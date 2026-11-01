import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from pathlib import Path


# ---------------------------------------------------------
# 1. 浅层空间注意力 (Spatial Attention)
# 作用：细化输入的语义掩码，解决边界伪影，提供软引导。
# ---------------------------------------------------------
class SpatialMaskAttention(nn.Module):
    def __init__(self, channels=1, kernel_size=3):
        super().__init__()
        if channels != 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("SAMR requires one mask channel and an odd positive kernel size")

        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mask):
        # mask: [B, 1, H, W]
        attn_map = self.sigmoid(self.conv(mask))
        return mask * attn_map


# ---------------------------------------------------------
# 2. 潜在全局自注意力 (Global Self-Attention)
# 作用：在 U-Net 瓶颈层捕捉长距离依赖，保证人脸全局结构协调。
# ---------------------------------------------------------
class LatentGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        if num_heads < 1 or dim < 1 or dim % num_heads:
            raise ValueError("LGSA channels must be divisible by a positive head count")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x input: [B, C, H, W] -> flatten to [B, N, C]
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        qkv = self.to_qkv(x_flat).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).to(v.dtype)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)

        # Restore shape: [B, C, H, W]
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------
# 3. 身份交叉注意力 (Identity Cross-Attention)
# 作用：将目标身份特征动态注入到生成过程，提升攻击迁移性。
# ---------------------------------------------------------
class IdentityCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        if min(query_dim, heads, dim_head) < 1:
            raise ValueError("DICA dimensions and head count must be positive")
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        # x: decoder queries [B, H*W, query_dim]
        # context: target identity tokens [B, N_identity, context_dim]
        if x.ndim != 3 or context.ndim != 3 or x.shape[0] != context.shape[0]:
            raise ValueError("DICA expects query/context token batches with matching batch sizes")
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(q.shape[0], -1, h, q.shape[-1] // h).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], -1, h, k.shape[-1] // h).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], -1, h, v.shape[-1] // h).permute(0, 2, 1, 3)

        dots = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1).to(v.dtype)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], -1, h * (out.shape[-1]))
        return self.to_out(out)


class HCAAttentionPipeline(nn.Module):
    """Checkpoint-backed inference wiring; no randomly initialized fallback.

    The checkpoint must describe its trained head counts, identity encoder and
    decoder blocks. Layer selection is not inferred from the complexity benchmark.
    Hooks are scoped to one sampling call and never installed by baseline entrypoints.
    """

    @staticmethod
    def load_checkpoint(path):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"HCA attention checkpoint not found: {path}. Full HCA inference requires "
                "trained SAMR/LGSA/DICA weights and configuration; these are not supplied "
                "with this repository. See README.md."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Expected HCA attention checkpoint format_version=1")
        if not {"config", "state_dict"}.issubset(checkpoint):
            raise ValueError("HCA attention checkpoint requires config and state_dict")
        return checkpoint

    def __init__(self, unet, checkpoint, surrogate_names):
        super().__init__()
        config = checkpoint["config"]
        self.identity_encoder = config["identity_encoder"]
        if self.identity_encoder not in surrogate_names:
            raise ValueError("HCA identity_encoder must be an available white-box surrogate, "
                             "not the held-out evaluation model")
        blocks = config["dica_blocks"]
        if (not isinstance(blocks, list) or not blocks
                or any(type(i) is not int or not 0 <= i < len(unet.output_blocks) for i in blocks)
                or len(set(blocks)) != len(blocks)):
            raise ValueError("dica_blocks must list distinct valid U-Net decoder block indices")
        identity_dim = config["identity_dim"]
        if type(identity_dim) is not int or identity_dim < 1:
            raise ValueError("identity_dim must be a positive integer")
        self.identity_dim = identity_dim
        self.samr = SpatialMaskAttention(kernel_size=config["samr_kernel_size"])
        self.lgsa = LatentGlobalAttention(unet.middle_block[-1].out_channels,
                                          config["lgsa_heads"])
        self.dicas = nn.ModuleDict({
            str(i): IdentityCrossAttention(
                unet.output_blocks[i][0].out_channels, identity_dim,
                heads=config["dica_heads"], dim_head=config["dica_dim_head"])
            for i in blocks
        })
        # Strict loading covers every projection and bias before any module is used.
        self.load_state_dict(checkpoint["state_dict"], strict=True)
        self.requires_grad_(False)
        self.eval()
        self._context = None
        self._handles = []

    def refine_mask(self, mask):
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("SAMR mask must have shape [B, 1, H, W]")
        weight = self.samr.conv.weight
        return self.samr(mask.to(device=weight.device, dtype=weight.dtype)).to(mask)

    @torch.no_grad()
    def target_identity(self, target, classifiers):
        # Same encoder input convention and cosine-normalized features as identity guidance.
        size = 160 if self.identity_encoder == "FaceNet" else 112
        encoder = classifiers[self.identity_encoder]
        weight = next(encoder.parameters())
        image = F.adaptive_avg_pool2d(target.to(device=weight.device, dtype=weight.dtype), size)
        feature = encoder(image).reshape(target.shape[0], -1)
        if feature.shape[1] != self.identity_dim:
            raise ValueError("Target identity dimension does not match the attention checkpoint")
        return F.normalize(feature, dim=-1).unsqueeze(1).detach()

    def _middle_hook(self, module, inputs, output):
        weight = self.lgsa.to_qkv.weight
        feature = output.to(device=weight.device, dtype=weight.dtype)
        return output + self.lgsa(feature).to(output)

    def _decoder_hook(self, key, output):
        if self._context is None:
            raise RuntimeError("DICA target identity is unset")
        batch, channels, height, width = output.shape
        context = self._context
        # DDIM concatenates [unconditional batch, conditional batch]. Both branches
        # need identity conditioning, including when text guidance scale is zero.
        if batch == 2 * context.shape[0]:
            context = context.repeat(2, 1, 1)
        elif batch != context.shape[0]:
            raise ValueError("DICA feature batch does not match target identity batch")
        attention = self.dicas[key]
        weight = attention.to_q.weight
        query = output.flatten(2).transpose(1, 2).to(device=weight.device, dtype=weight.dtype)
        context = context.to(device=weight.device, dtype=weight.dtype)
        update = attention(query, context).transpose(1, 2).reshape(batch, channels, height, width)
        return output + update.to(output)

    @contextmanager
    def denoising(self, unet, identity):
        if self._handles:
            raise RuntimeError("HCA sampling contexts cannot be nested")
        self._context = identity.detach()
        try:
            self._handles.append(unet.middle_block.register_forward_hook(self._middle_hook))
            for key in self.dicas:
                def hook(module, inputs, output, key=key):
                    return self._decoder_hook(key, output)
                self._handles.append(unet.output_blocks[int(key)].register_forward_hook(hook))
            yield
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._context = None
