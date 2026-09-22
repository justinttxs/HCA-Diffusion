# Third-Party Notices

This repository contains or builds upon code and models from several open-source projects.

## Adv-Diffusion

HCA-Diffusion is implemented based on the Adv-Diffusion codebase:

Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model
Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruimin Hu, and Xinbo Gao
AAAI 2024

The original Adv-Diffusion repository is distributed under the MIT License.

The original copyright notice in the root `LICENSE` file is retained.

## Latent Diffusion

Parts of the `ldm/` directory are derived from or based on the CompVis latent-diffusion / Stable Diffusion codebase and related open-source diffusion implementations.

The CompVis latent-diffusion project is distributed under the MIT License.

Some individual source files under `ldm/` also contain attribution comments referring to upstream projects such as:

- OpenAI guided-diffusion
- OpenAI improved-diffusion
- CompVis taming-transformers
- MiDaS
- k-diffusion
- BasicSR

The corresponding upstream licenses and attribution requirements continue to apply to those portions.

## Face Parsing

The face-parsing components used by the original Adv-Diffusion implementation are based on the open-source face-parsing.PyTorch project.

That upstream project is distributed under the MIT License.

The pretrained face-parsing checkpoint is not distributed in this repository.

## Face Recognition Models

The original Adv-Diffusion implementation uses the following face-recognition models:

- IR152
- IRSE50
- FaceNet
- MobileFace

The upstream Adv-Diffusion repository refers users to AMT-GAN for the corresponding pretrained checkpoints.

These pretrained face-recognition checkpoints are not distributed in this repository.

Users should consult the corresponding upstream projects for checkpoint licenses, usage conditions, and attribution requirements.

## Pretrained Models and Datasets

This repository does not redistribute:

- Stable Diffusion checkpoints
- Face-recognition checkpoints
- Face-parsing checkpoints
- CelebA-HQ
- FFHQ
- LFW
- CFP-FP

Users are responsible for obtaining these resources from their respective official or upstream sources and complying with their applicable licenses and terms.

## License Scope

The root `LICENSE` file is retained from the original Adv-Diffusion codebase.

Third-party code, pretrained models, and datasets remain subject to their own licenses, copyright notices, and terms of use.
