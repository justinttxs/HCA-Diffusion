# HCA-Diffusion

Official implementation of **HCA-Diffusion**, a diffusion-based adversarial attack framework for face recognition.

HCA-Diffusion introduces a hierarchical attention design into the diffusion-based adversarial generation process. The framework contains three main components:

- **SAMR (Spatial Attention Mask Refinement)** applies convolutional attention to the Face Parsing mask, providing a soft spatial condition for inpainting.
- **LGSA (Latent Global Self-Attention)** aggregates spatial tokens at the U-Net bottleneck and adds the projected attention features back into the denoising stream to model global structure.
- **DICA (Dynamic Identity Cross-Attention)** uses decoder features as queries and a target face identity embedding as key/value context. Its projected output is added to decoder features to provide identity conditioning during denoising.

This repository contains the HCA-Diffusion inference wiring together with scripts for multi-seed evaluation, hyperparameter sensitivity analysis, Adv-Diffusion baseline evaluation, and computational complexity analysis. Full HCA inference requires trained SAMR/LGSA/DICA weights and their matching configuration. These attention weights and an authoritative trained layer configuration are not distributed here; the base diffusion and face-recognition checkpoints alone are insufficient.

### Generation Pipeline

`eval.py` reads source/target face pairs and parses the source into semantic regions. The selected labels are skin, nose, glasses, eyes, eyebrows, mouth and lips (`1, 2, 3, 4, 5, 6, 7, 10, 11, 12`). The existing mask convention is retained: these selected regions are zero and their complement is one. The binary mask determines the masked source image; SAMR refines the complementary mask without extending its support. Area interpolation resizes the soft mask from image resolution to latent resolution (512 × 512 to 64 × 64), and the result is concatenated with the encoded masked image as the U-Net spatial condition.

The source image is encoded and noised at timestep `T`. During each reverse DDIM step, LGSA updates the U-Net middle-block output before decoding. DICA updates the outputs of the decoder blocks declared by the attention checkpoint. The identity encoder is reused from the existing white-box surrogates, with the same image normalization and pooling convention as classifier guidance; its normalized target embedding supplies one identity token. The held-out recognition model is used only for evaluation. Both internally concatenated classifier-free guidance batches receive the matching identity context, including with the existing text guidance scale of zero.

The original cosine-similarity classifier guidance, latent sampling and image decoding remain in use. Generated images and ASR results are saved under `--save`. `eval_source.py` runs the independent Adv-Diffusion baseline without loading or installing HCA attention. The integration reuses the existing attention equations and adds no training procedure. With a single identity token, DICA's softmax has one key, so spatially varying identity attention must not be inferred from this implementation.

---

## 1. Environment

The experiments were conducted with the following environment:

- Python 3.9.16
- PyTorch 2.0.0
- CUDA 11.8

Create the Conda environment using:

```bash
conda env create -f env.yaml
conda activate hca-diffusion
```

---

## 2. Repository Structure

```text
HCA/
├── README.md
├── LICENSE
├── env.yaml
├── .gitignore
│
├── eval.py
├── eval_source.py
├── attention_modules.py
├── benchmark_hca_modules.py
├── classifier.py
├── dataset.py
├── psnr_ssim.py
├── utils.py
│
├── configs/
├── FaceParsing/
├── fr_model/
├── ldm/
│
├── scripts/
│   ├── run_hca_5seeds.sh
│   ├── run_sensitivity_ts_ffhq_mobile.sh
│   └── run_advdiffusion_ffhq_seed4_t50.sh
│
└── pretrained_model/        # not included in this repository
```

The `pretrained_model/` directory, datasets, generated adversarial images, logs, and raw experimental outputs are excluded from the repository.

---

## 3. Pretrained Models

Pretrained checkpoints are not included in this repository.

Please create the following directory structure:

```text
pretrained_model/
├── 512-inpainting-ema.ckpt
├── ir152.pth
├── irse50.pth
├── mobile_face.pth
├── facenet.pth
├── 38_G.pth
├── hca_attention.pt          # required separately for full HCA inference
└── open_clip/
    └── open_clip_pytorch_model.bin
```

### Stable Diffusion Inpainting Model

The Stable Diffusion inpainting checkpoint should be placed at:

```text
pretrained_model/512-inpainting-ema.ckpt
```

### Face Recognition Models

The face-recognition checkpoints should be placed at:

```text
pretrained_model/ir152.pth
pretrained_model/irse50.pth
pretrained_model/mobile_face.pth
pretrained_model/facenet.pth
```

The supported face-recognition architectures are:

- IR152
- IRSE50
- FaceNet
- MobileFace

### Face Parsing Model

The face-parsing checkpoint should be placed at:

```text
pretrained_model/38_G.pth
```

### OpenCLIP Model

The OpenCLIP checkpoint should be placed at:

```text
pretrained_model/open_clip/open_clip_pytorch_model.bin
```

### HCA Attention Checkpoint

`eval.py` requires `pretrained_model/hca_attention.pt`, or a path supplied with `--attention-checkpoint`. It stops if this file is absent; there is no random-weight or zero-output fallback. A compatible checkpoint must contain trained weights for the exact residual integration described above. Renaming a base Stable Diffusion checkpoint or saving newly initialized attention modules does not satisfy this requirement.

The loader accepts a tensor/primitive dictionary with this interface:

```text
format_version: 1
config:
  identity_encoder: name of an available white-box FR model
  identity_dim: output embedding width of that encoder
  samr_kernel_size: positive odd convolution kernel size
  lgsa_heads: head count dividing the U-Net bottleneck width
  dica_blocks: nonempty list of distinct U-Net output_blocks indices
  dica_heads: positive cross-attention head count
  dica_dim_head: positive cross-attention head width
state_dict:
  samr.*: SpatialMaskAttention parameters
  lgsa.*: LatentGlobalAttention parameters
  dicas.<block_index>.*: IdentityCrossAttention parameters for each selected block
```

This is the inference loader's compatibility contract, not a supplied trained artifact or a claim about unpublished paper settings. Bottleneck/decoder channel widths are taken from the loaded U-Net; identity context width and head counts come from the checkpoint configuration. Every attention parameter is loaded strictly before use. The declared identity encoder must remain a surrogate for the selected `--model`; checkpoints for different held-out settings may therefore require different encoder configurations and matching weights. Existing attention weights in a different naming format must be mapped explicitly after confirming architecture and residual semantics.

The two HCA shell scripts accept `ATTENTION_CHECKPOINT` to select this artifact. Keep the checkpoint and configuration fixed across seeds and sensitivity runs. The attention checkpoint, like all other model weights, is excluded from Git.

---

## 4. Dataset Preparation

The main experiments use **CelebA-HQ** and **FFHQ**.

Prepare the datasets using the following directory structure:

```text
celeba-hq_sample/
├── src/
└── target/

ffhq_sample/
├── src/
└── target/
```

Here:

- `src/` contains source face images.
- `target/` contains target-identity face images.

For example:

```text
ffhq_sample/
├── src/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── target/
    ├── 00000.png
    ├── 00001.png
    └── ...
```

Dataset files are not distributed with this repository.

---

## 5. Running HCA-Diffusion

After providing a compatible trained attention checkpoint, a single HCA-Diffusion experiment can be launched with:

```bash
python eval.py \
    --model IR152 \
    --dataset celeba \
    --num 92 \
    --t 50 \
    --s 300 \
    --seed 4 \
    --save results/celeba_IR152_seed4
```

The main command-line arguments are:

| Argument | Description |
|---|---|
| `--model` | Held-out target face-recognition model |
| `--dataset` | Evaluation dataset |
| `--num` | Number of evaluated source images |
| `--t` | Diffusion timestep |
| `--s` | Classifier guidance strength |
| `--seed` | Random seed |
| `--save` | Output directory |
| `--attention-checkpoint` | Trained attention checkpoint and matching configuration (HCA only) |

The main experimental configuration uses:

```text
T = 50
s = 300
```

unless otherwise specified.

---

## 6. Target Face Recognition Models

The evaluation considers four face-recognition architectures:

```text
IR152
IRSE50
FaceNet
MobileFace
```

For each held-out target model, the remaining face-recognition models are used as surrogate models during adversarial generation.

For example:

```bash
python eval.py \
    --model FaceNet \
    --dataset ffhq \
    --num 92 \
    --t 50 \
    --s 300 \
    --seed 4 \
    --save results/ffhq_FaceNet_seed4
```

---

## 7. Multi-Seed Evaluation

The reported repeated experiments use five random seeds:

```text
4, 5, 6, 7, 8
```

Run the five-seed evaluation with:

```bash
bash scripts/run_hca_5seeds.sh
```

The default configuration can be overridden using environment variables.

For example:

```bash
DATASET=ffhq \
MODEL=IR152 \
NUM=92 \
T=50 \
S=300 \
bash scripts/run_hca_5seeds.sh
```

Another example for FaceNet is:

```bash
DATASET=ffhq \
MODEL=FaceNet \
NUM=92 \
T=50 \
S=300 \
bash scripts/run_hca_5seeds.sh
```

The script creates separate output directories for each random seed.

---

## 8. Hyperparameter Sensitivity

The repository includes a script for evaluating sensitivity to the diffusion timestep `T` and classifier guidance strength `s`.

Run:

```bash
bash scripts/run_sensitivity_ts_ffhq_mobile.sh
```

The current configuration evaluates:

```text
T = 35, 50, 65
s = 200, 300, 400
```

using:

```text
Dataset: FFHQ
Target model: MobileFace
Seed: 4
```

The default setting is:

```text
T = 50
s = 300
```

---

## 9. Adv-Diffusion Baseline

`eval_source.py` is used for the corresponding Adv-Diffusion baseline evaluation.

A single baseline experiment can be run with:

```bash
python eval_source.py \
    --model IR152 \
    --dataset ffhq \
    --num 92 \
    --t 50 \
    --s 300 \
    --seed 4 \
    --save res_source/ffhq_IR152_seed4
```

For the FFHQ seed-4 evaluation across all four held-out face-recognition models, run:

```bash
bash scripts/run_advdiffusion_ffhq_seed4_t50.sh
```

The script evaluates:

```text
IR152
IRSE50
FaceNet
MobileFace
```

with:

```text
Dataset: FFHQ
Seed: 4
T = 50
s = 300
```

---

## 10. Computational Complexity

The existing complexity-only attention harness can be run using:

```bash
python benchmark_hca_modules.py \
    --configs baseline,samr,lgsa,dica,full \
    --dataset ffhq \
    --model IR152 \
    --num 10 \
    --batch-size 2 \
    --warmup-batches 1 \
    --seed 4 \
    --t 50 \
    --s 300 \
    --ddim-steps 45 \
    --num-heads 8 \
    --dim-head 64 \
    --dica-blocks all \
    --profile-flops \
    --out-dir complexity_hca_modules
```

Available configurations are:

```text
baseline
samr
lgsa
dica
full
```

where:

- `baseline` evaluates the diffusion pipeline without the three HCA attention components.
- `samr` executes the SAMR computation.
- `lgsa` executes a bottleneck LGSA hook.
- `dica` executes DICA hooks on selected decoder blocks.
- `full` executes all three attention computations.

The complexity script is intended to measure computational properties such as:

- runtime
- GPU memory consumption
- additional parameters
- FLOPs

This harness uses temporary, untrained modules: LGSA/DICA output projections are zero-initialized and SAMR's output is excluded from the mask update. It does not load the trained HCA attention checkpoint and does not validate attack effectiveness. Its measurements describe the configured benchmark wiring, which may differ from the decoder blocks or head dimensions of a trained HCA checkpoint. Use `eval.py` for checkpoint-backed HCA generation.

---

## 11. Evaluation Metrics

The experiments use attack-effectiveness and image-quality metrics including:

- Attack Success Rate (ASR)
- FID
- PSNR
- SSIM
- LPIPS

The main evaluation code reports attack-related results, while image-quality metrics can be computed separately from generated adversarial examples.

PSNR and SSIM can be evaluated using:

```bash
python psnr_ssim.py \
    --dir0 <adversarial_image_directory> \
    --dir1 <source_image_directory>
```

---

## 12. Output Files

A typical experiment saves generated results under the directory specified by `--save`.

For example:

```text
results/
└── ffhq_IR152_seed4/
    └── ...
```

Generated images and raw experimental outputs are ignored by Git and are not included in this repository.

This prevents datasets, pretrained checkpoints, generated face images, and large experiment artifacts from being accidentally uploaded.

---

## 13. Reproducibility

For consistent reproduction of the reported experiments, please keep the following settings fixed when performing direct comparisons:

```text
Image resolution: 512 x 512
T: 50
s: 300
Random seeds: 4, 5, 6, 7, 8
Target models: IR152, IRSE50, FaceNet, MobileFace
```

The exact dataset split, source images, target images, pretrained checkpoints, and random seeds should remain consistent across compared methods.

---

## 14. Additional Datasets

The evaluation code also contains support for additional face datasets when the corresponding directory structure is prepared.

For example, `eval_source.py` accepts:

```text
celeba
ffhq
lfw
cfp
```

The main experiments in the paper focus on CelebA-HQ and FFHQ.

---

## 15. Notes

- Pretrained model weights are not included.
- Dataset images are not included.
- Generated adversarial face images are not included.
- Raw experiment logs are not included.
- Temporary and machine-specific files are excluded through `.gitignore`.
- Experiment outputs should be stored locally rather than committed to the repository.

---

## 16. Acknowledgements

This implementation builds upon diffusion-based adversarial generation and Stable Diffusion code components.

The repository retains the license and copyright notice associated with the original codebase.

Please also refer to the licenses and terms of the corresponding third-party models and datasets when using their pretrained checkpoints or data.

---

## 17. License

This repository retains the MIT License included with the original codebase.

See [LICENSE](LICENSE) for the complete license text. Third-party components included in this repository remain subject to their respective licenses and attribution requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for additional attribution information.

---

## 18. Citation

Citation information will be updated after publication.
