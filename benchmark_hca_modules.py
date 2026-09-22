#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complexity-only runtime integration of SAMR + LGSA + DICA.

Does not modify eval.py/eval_source.py, does not save adversarial images, and does not
write result.json. Outputs only to --out-dir.

Configs: baseline, samr, lgsa, dica, full.
LGSA is hooked after U-Net middle_block. DICA is hooked after selected decoder
output_blocks. Decoder features are DICA queries; a target-identity feature token
from an already-loaded surrogate is the key/value context.

LGSA/DICA output projections are zero-initialized only to keep the untrained
complexity-only wiring numerically stable. Their computations, FLOPs, parameters,
latency and memory allocation still occur. This script is for complexity measurement,
not for claiming ASR of the newly wired pipeline.
"""

import argparse, csv, gc, json, time
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from utils import setup_seed, get_fr_model, initialize_model
from dataset import base_dataset
from FaceParsing.interface import FaceParsing
from attention_modules import SpatialMaskAttention, LatentGlobalAttention, IdentityCrossAttention

CONFIGS = ("baseline", "samr", "lgsa", "dica", "full")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gib(v):
    return float(v) / (1024 ** 3)


def pct(new, base):
    if new is None or base in (None, 0):
        return None
    return (new - base) / base * 100.0


def get_dataset(name, transform):
    if name == "ffhq":
        return base_dataset(dir="./ffhq_sample", transform=transform)
    if name == "celeba":
        return base_dataset(dir="./celeba-hq_sample", transform=transform)
    if name == "lfw":
        return base_dataset(dir="./lfw_sample", transform=transform)
    raise ValueError(name)


class AttentionInjector:
    def __init__(self, unet, use_lgsa=False, use_dica=False, dica_blocks="all",
                 num_heads=8, dim_head=64):
        self.unet = unet
        self.use_lgsa = use_lgsa
        self.use_dica = use_dica
        self.dica_blocks_spec = dica_blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.handles = []
        self.lgsa = None
        self.dicas = {}
        self.identity_context = None
        self.reset_counters()
        self._install()

    def reset_counters(self):
        self.calls = {"lgsa": 0, "dica": 0}
        self.flops = {"lgsa": 0.0, "dica": 0.0}
        self.shapes = {"lgsa": [], "dica": []}

    def _selected_blocks(self):
        n = len(self.unet.output_blocks)
        if self.dica_blocks_spec == "all":
            return list(range(n))
        ids = sorted(set(int(x.strip()) for x in self.dica_blocks_spec.split(",") if x.strip()))
        if not ids or min(ids) < 0 or max(ids) >= n:
            raise ValueError(f"Bad --dica-blocks; valid range is 0..{n-1}")
        return ids

    @staticmethod
    def _zero_output(m):
        with torch.no_grad():
            if hasattr(m, "proj"):
                m.proj.weight.zero_()
                if m.proj.bias is not None:
                    m.proj.bias.zero_()
            if hasattr(m, "to_out"):
                m.to_out.weight.zero_()
                if m.to_out.bias is not None:
                    m.to_out.bias.zero_()

    @staticmethod
    def lgsa_flops(B, C, H, W):
        # MAC -> 2 FLOPs convention; excludes softmax/reshape elementwise ops.
        N = H * W
        return (2*B*N*C*(3*C) +       # qkv
                2*B*N*N*C +           # QK^T
                2*B*N*N*C +           # Attn*V
                2*B*N*C*C)            # output projection

    @staticmethod
    def dica_flops(B, Nq, Cq, Nk, Cctx, inner):
        return (2*B*Nq*Cq*inner +      # q
                2*B*Nk*Cctx*inner +   # k
                2*B*Nk*Cctx*inner +   # v
                2*B*Nq*Nk*inner +     # QK^T
                2*B*Nq*Nk*inner +     # Attn*V
                2*B*Nq*inner*Cq)       # output projection

    def _install(self):
        if self.use_lgsa:
            self.handles.append(self.unet.middle_block.register_forward_hook(self._middle_hook))
        if self.use_dica:
            for idx in self._selected_blocks():
                def make_hook(i):
                    def hook(module, inputs, output):
                        return self._decoder_hook(i, output)
                    return hook
                self.handles.append(self.unet.output_blocks[idx].register_forward_hook(make_hook(idx)))

    def set_identity_context(self, x):
        self.identity_context = x.detach()

    def _middle_hook(self, module, inputs, output):
        if not torch.is_tensor(output) or output.ndim != 4:
            return output
        B, C, H, W = output.shape
        if self.lgsa is None:
            if C % self.num_heads != 0:
                raise RuntimeError(f"LGSA C={C} is not divisible by heads={self.num_heads}")
            self.lgsa = LatentGlobalAttention(C, self.num_heads).to(output.device, output.dtype).eval()
            self._zero_output(self.lgsa)
        y = self.lgsa(output)
        self.calls["lgsa"] += 1
        self.flops["lgsa"] += self.lgsa_flops(B, C, H, W)
        shp = [int(B), int(C), int(H), int(W)]
        if shp not in self.shapes["lgsa"]:
            self.shapes["lgsa"].append(shp)
        return output + y

    def _decoder_hook(self, idx, output):
        if not torch.is_tensor(output) or output.ndim != 4:
            return output
        if self.identity_context is None:
            raise RuntimeError("DICA enabled but identity context is unset")
        B, C, H, W = output.shape
        ctx = self.identity_context
        # Stable-Diffusion classifier-free guidance may concatenate conditional and
        # unconditional batches inside the U-Net (e.g. feature batch = 2 * image batch).
        # Repeat the target-identity token to match that internal batch expansion.
        if ctx.shape[0] != B:
            if B % ctx.shape[0] != 0:
                raise RuntimeError(f"DICA batch mismatch: feature={B}, identity={ctx.shape[0]}")
            ctx = ctx.repeat(B // ctx.shape[0], 1, 1)
        Cctx = int(ctx.shape[-1])
        key = str(idx)
        if key not in self.dicas:
            m = IdentityCrossAttention(
                query_dim=C,
                context_dim=Cctx,
                heads=self.num_heads,
                dim_head=self.dim_head,
            ).to(output.device, output.dtype).eval()
            self._zero_output(m)
            self.dicas[key] = m
        m = self.dicas[key]
        q = output.permute(0, 2, 3, 1).reshape(B, H*W, C)
        ctx = ctx.to(output.device, output.dtype)
        y = m(q, ctx)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        self.calls["dica"] += 1
        self.flops["dica"] += self.dica_flops(
            B, H*W, C, int(ctx.shape[1]), Cctx, self.num_heads*self.dim_head
        )
        shp = [int(idx), int(B), int(C), int(H), int(W), int(Cctx)]
        if shp not in self.shapes["dica"]:
            self.shapes["dica"].append(shp)
        return output + y

    def extra_params(self):
        n = 0
        if self.lgsa is not None:
            n += sum(p.numel() for p in self.lgsa.parameters())
        for m in self.dicas.values():
            n += sum(p.numel() for p in m.parameters())
        return int(n)

    def analytical_flops(self):
        return float(self.flops["lgsa"] + self.flops["dica"])

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def prepare(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = get_dataset(args.dataset, tfm)
    if args.num > len(ds):
        raise ValueError(f"--num={args.num} exceeds dataset length={len(ds)}")
    loader = DataLoader(Subset(ds, list(range(args.num))), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, drop_last=False)

    sampler = initialize_model(
        "configs/stable-diffusion/v2-inpainting-inference.yaml",
        "pretrained_model/512-inpainting-ema.ckpt"
    )
    model = sampler.model
    unet = model.model.diffusion_model

    names = ["IR152", "IRSE50", "FaceNet", "MobileFace"]
    whitebox = [x for x in names if x != args.model]
    classifier = {n: get_fr_model(n) for n in whitebox}
    for m in classifier.values():
        m.eval()
    return device, loader, sampler, model, unet, classifier, whitebox


def identity_token(tgt_image, classifier):
    # Reuse the first already-loaded surrogate; no extra FR model is loaded.
    name, encoder = next(iter(classifier.items()))
    size = 160 if name == "FaceNet" else 112
    resize = nn.AdaptiveAvgPool2d((size, size))
    with torch.no_grad():
        f = encoder(resize(tgt_image)).reshape(tgt_image.shape[0], -1)
        f = torch.nn.functional.normalize(f, dim=1)
    return f.unsqueeze(1), name


def samr_flops(num_images, H=512, W=512):
    # Conv2d 1->1, 3x3 only; MAC -> 2 FLOPs. Sigmoid/elementwise omitted.
    return 2.0 * num_images * H * W * 3 * 3


@torch.no_grad()
def generate(config, image, tgt_image, ctx, args, injector=None):
    device, _, sampler, model, unet, classifier, _ = ctx
    h = w = 512
    B = image.shape[0]
    txt = ""

    # Same face-parsing stage as current eval.py.
    face_parsing = FaceParsing()
    pred = face_parsing(image)
    mask = None
    for label in [1,2,3,4,5,6,7,10,11,12]:
        cur = pred == label
        mask = cur if mask is None else (mask | cur)
    mask = (mask == 0).float().reshape(B, 1, h, w).to(device)
    image = image.to(device)
    tgt_image = tgt_image.to(device)

    if config in ("samr", "full"):
        samr = SpatialMaskAttention(1, 3).to(device=device, dtype=mask.dtype).eval()
        # Execute actual SAMR compute without altering the mask in this complexity-only run.
        refined = samr(mask)
        mask = mask + 0.0 * refined

    identity_encoder = None
    if config in ("dica", "full"):
        tok, identity_encoder = identity_token(tgt_image, classifier)
        injector.set_identity_context(tok)

    masked_image = image * (mask < 0.5)
    batch = {"image": image, "txt": B*[txt], "mask": mask, "masked_image": masked_image}

    c = model.cond_stage_model.encode(batch["txt"])
    c_cat = []
    for ck in model.concat_keys:
        cc = batch[ck].float()
        if ck != model.masked_image_key:
            cc = torch.nn.functional.interpolate(cc, size=[h//8, w//8])
        else:
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc.to(device)))
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)

    cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    uc_cross = model.get_unconditional_conditioning(B, "")
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

    shape = [model.channels, h//8, w//8]
    z = model.get_first_stage_encoding(model.encode_first_stage(image))
    t = torch.tensor([args.t]*B, device=device)
    z_t = model.q_sample(x_start=z, t=t)

    samples_cfg, _ = sampler.sample(
        args.ddim_steps, B, shape, cond,
        verbose=False, eta=1.0,
        unconditional_guidance_scale=0,
        unconditional_conditioning=uc_full,
        x_T=z_t, _t=args.t + 1, log_every_t=1,
        classifier=classifier,
        classifier_scale=args.s,
        x_target=tgt_image,
    )
    result = torch.clamp(model.decode_first_stage(samples_cfg), -1, 1)
    return result, identity_encoder


def make_injector(config, unet, args):
    return AttentionInjector(
        unet,
        use_lgsa=config in ("lgsa", "full"),
        use_dica=config in ("dica", "full"),
        dica_blocks=args.dica_blocks,
        num_heads=args.num_heads,
        dim_head=args.dim_head,
    )


def benchmark_config(config, ctx, args):
    device, loader, _, _, unet, _, whitebox = ctx
    inj = make_injector(config, unet, args)

    # Warm-up on a temporary iterator; measured pass restarts from image 0.
    it = iter(loader)
    for _ in range(args.warmup_batches):
        try:
            image, tgt = next(it)
        except StopIteration:
            break
        out, _ = generate(config, image, tgt.to(device), ctx, args, inj)
        sync(); del out

    inj.reset_counters()
    times, peak_a, peak_r = [], [], []
    nimg = 0
    identity_encoder = None

    for bi, (image, tgt) in enumerate(loader):
        B = image.shape[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        out, enc = generate(config, image, tgt.to(device), ctx, args, inj)
        sync(); dt = time.perf_counter() - t0
        if enc is not None:
            identity_encoder = enc
        times.extend([dt/B]*B); nimg += B
        if torch.cuda.is_available():
            peak_a.append(gib(torch.cuda.max_memory_allocated()))
            peak_r.append(gib(torch.cuda.max_memory_reserved()))
        else:
            peak_a.append(0.0); peak_r.append(0.0)
        print(f"[{config}] batch={bi:03d} B={B} time/img={dt/B:.4f}s peak={peak_a[-1]:.3f}GB")
        del out

    attn_flops = inj.analytical_flops()
    if config in ("samr", "full"):
        attn_flops += samr_flops(nimg)

    row = {
        "config": config,
        "measured_images": nimg,
        "time_s_per_image_mean": mean(times),
        "time_s_per_image_std": stdev(times) if len(times) > 1 else 0.0,
        "peak_allocated_gb": max(peak_a) if peak_a else 0.0,
        "peak_reserved_gb": max(peak_r) if peak_r else 0.0,
        "analytical_attention_tflops_per_image": attn_flops/nimg/1e12 if nimg else None,
        "extra_attention_params": inj.extra_params(),
        "lgsa_calls_measured": inj.calls["lgsa"],
        "dica_calls_measured": inj.calls["dica"],
        "lgsa_shapes": inj.shapes["lgsa"],
        "dica_shapes": inj.shapes["dica"],
        "identity_encoder_for_dica": identity_encoder,
        "whitebox_surrogates": whitebox,
    }
    inj.close(); del inj; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row


def profile_one_batch(config, ctx, args):
    device, loader, _, _, unet, _, _ = ctx
    image, tgt = next(iter(loader))
    B = image.shape[0]
    inj = make_injector(config, unet, args)
    inj.reset_counters()

    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available(): acts.append(torch.profiler.ProfilerActivity.CUDA)
    sync()
    with torch.profiler.profile(activities=acts, with_flops=True,
                                record_shapes=False, profile_memory=False,
                                with_stack=False) as prof:
        out, _ = generate(config, image, tgt.to(device), ctx, args, inj)
        sync()
    total = 0.0; events = 0
    for e in prof.key_averages():
        f = getattr(e, "flops", 0)
        if f:
            total += float(f); events += 1
    analytical = inj.analytical_flops()
    if config in ("samr", "full"):
        analytical += samr_flops(B)
    meta = {
        "profiled_tflops_per_image": total/B/1e12,
        "profiled_flop_events": events,
        "profile_batch_attention_tflops_per_image": analytical/B/1e12,
    }
    del out
    inj.close(); del inj; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="baseline,samr,lgsa,dica,full")
    p.add_argument("--dataset", choices=["ffhq","celeba","lfw"], default="ffhq")
    p.add_argument("--model", choices=["IR152","IRSE50","FaceNet","MobileFace"], default="IR152")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--t", type=int, default=50)
    p.add_argument("--s", type=float, default=300)
    p.add_argument("--ddim-steps", type=int, default=45)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=64)
    p.add_argument("--dica-blocks", default="all", help="all or comma list, e.g. 0,1,2")
    p.add_argument("--profile-flops", action="store_true")
    p.add_argument("--out-dir", default="complexity_hca_modules")
    args = p.parse_args()

    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    if any(x not in CONFIGS for x in configs):
        raise ValueError(f"configs must be subset of {CONFIGS}")

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("="*80)
    print("HCA MODULE COMPLEXITY BENCHMARK")
    print(f"configs={configs} dataset={args.dataset} heldout={args.model}")
    print(f"num={args.num} batch={args.batch_size} DDIM={args.ddim_steps} t={args.t} s={args.s}")
    print(f"DICA blocks={args.dica_blocks} GPU={gpu}")
    print("="*80)

    ctx = prepare(args)
    rows = []
    for config in configs:
        print("\n" + "#"*80 + f"\n# {config}\n" + "#"*80)
        setup_seed(args.seed)
        row = benchmark_config(config, ctx, args)
        if args.profile_flops:
            setup_seed(args.seed)
            row.update(profile_one_batch(config, ctx, args))
        else:
            row.update({"profiled_tflops_per_image": None,
                        "profiled_flop_events": None,
                        "profile_batch_attention_tflops_per_image": None})
        row.update({"gpu": gpu, "dataset": args.dataset, "heldout_model": args.model,
                    "batch_size": args.batch_size, "ddim_steps": args.ddim_steps,
                    "ddpm_start_t": args.t, "guidance_s": args.s,
                    "dica_blocks": args.dica_blocks})
        rows.append(row)
        (outdir/f"{config}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    base = next((r for r in rows if r["config"] == "baseline"), None)
    for r in rows:
        if base:
            r["time_overhead_pct_vs_baseline"] = pct(r["time_s_per_image_mean"], base["time_s_per_image_mean"])
            r["memory_overhead_pct_vs_baseline"] = pct(r["peak_allocated_gb"], base["peak_allocated_gb"])
            r["profiled_flops_overhead_pct_vs_baseline"] = pct(r["profiled_tflops_per_image"], base["profiled_tflops_per_image"])
        else:
            r["time_overhead_pct_vs_baseline"] = None
            r["memory_overhead_pct_vs_baseline"] = None
            r["profiled_flops_overhead_pct_vs_baseline"] = None

    fields = [
        "config","time_s_per_image_mean","time_s_per_image_std",
        "peak_allocated_gb","peak_reserved_gb","profiled_tflops_per_image",
        "analytical_attention_tflops_per_image","extra_attention_params",
        "time_overhead_pct_vs_baseline","memory_overhead_pct_vs_baseline",
        "profiled_flops_overhead_pct_vs_baseline","measured_images","gpu",
        "dataset","heldout_model","batch_size","ddim_steps","ddpm_start_t",
        "guidance_s","dica_blocks"
    ]
    with (outdir/"complexity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fields})
    (outdir/"complexity_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def fmt(v, n=3): return "N/A" if v is None else f"{v:.{n}f}"
    print("\n" + "="*120)
    print("FINAL COMPLEXITY SUMMARY")
    print("="*120)
    print(f"{'Config':<10}{'Time(s/img)':>14}{'PeakGB':>12}{'Prof.TF/img':>16}{'Attn.TF/img':>16}{'ΔTime%':>12}{'ΔMem%':>12}{'ΔFLOPs%':>12}")
    print("-"*120)
    for r in rows:
        print(f"{r['config']:<10}{fmt(r['time_s_per_image_mean']):>14}{fmt(r['peak_allocated_gb']):>12}{fmt(r['profiled_tflops_per_image']):>16}{fmt(r['analytical_attention_tflops_per_image'],6):>16}{fmt(r['time_overhead_pct_vs_baseline'],2):>12}{fmt(r['memory_overhead_pct_vs_baseline'],2):>12}{fmt(r['profiled_flops_overhead_pct_vs_baseline'],2):>12}")
    print("="*120)
    print(f"CSV : {outdir/'complexity_summary.csv'}")
    print(f"JSON: {outdir/'complexity_summary.json'}")
    print("No adversarial images and no ASR result.json are written.")
    print("This is a complexity-only integration; ASR must be rerun before claiming performance for this wiring.")


if __name__ == "__main__":
    main()
