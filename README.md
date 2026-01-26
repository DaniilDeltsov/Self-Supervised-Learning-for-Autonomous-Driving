# Self-Supervised Learning for Autonomous Driving

Master's thesis. Comparing self-supervised pre-training strategies for semantic
segmentation on Cityscapes.

## Question

Do self-supervised pre-training strategies produce features that transfer better to
urban-scene semantic segmentation than supervised baselines, and which method wins?

## Method

A controlled comparison isolating pre-training as the only varying factor: random,
ImageNet-supervised, MoCo v3 and DINO backbones, each as a linear probe and a full
fine-tune, with an FCN head and cross-entropy loss.

## Results

Cityscapes val mIoU (%)

| Backbone | Pre-training | Linear probe | Full fine-tune |
|---|---|:---:|:---:|
| ResNet-50 | random init | — | 34.40 |
| ResNet-50 | ImageNet supervised | 43.27 | 53.62 |
| ResNet-50 | MoCo v3 | 43.61 | 54.69 |
| ViT-S/16  | DINO | 43.66 | 55.26 |

## Repository

- `cityscapes_eda.py` — exploratory data analysis
- `model_experiments.ipynb` — controlled backbone comparison
