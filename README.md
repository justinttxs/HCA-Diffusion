# HCA-Diffusion

Official implementation of **HCA-Diffusion**, a diffusion-based adversarial attack framework for face recognition.

HCA-Diffusion introduces a hierarchical attention design into the diffusion-based adversarial generation process. The framework contains three main components:

- **SAMR**: Spatial Attention Mask Refinement
- **LGSA**: Latent Global Self-Attention
- **DICA**: Decoder Identity Cross-Attention

This repository contains the implementation of HCA-Diffusion together with scripts for multi-seed evaluation, hyperparameter sensitivity analysis, Adv-Diffusion baseline evaluation, and computational complexity analysis.

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

A single HCA-Diffusion experiment can be launched with:

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

The computational overhead of the hierarchical attention components can be evaluated using:

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
- `samr` enables SAMR.
- `lgsa` enables LGSA.
- `dica` enables DICA.
- `full` enables the complete hierarchical attention configuration.

The complexity script is intended to measure computational properties such as:

- runtime
- GPU memory consumption
- additional parameters
- FLOPs

The LGSA and DICA integrations in this script are used for computational complexity measurement.

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
