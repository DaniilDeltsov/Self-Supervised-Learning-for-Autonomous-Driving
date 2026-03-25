# Self-Supervised Learning for Autonomous Driving

Master's thesis. DINO wins the pre-training comparison, so the second experiment pushes
a DINO ViT-S/16 backbone to its practical mIoU ceiling on Cityscapes.

## Question

Do self-supervised pre-training strategies produce features that transfer better to
urban-scene semantic segmentation than supervised baselines, and which method wins?

## Method

Two experiments. A controlled comparison isolating pre-training (random, ImageNet
supervised, MoCo v3 and DINO backbones; linear probe + full fine-tune; FCN head;
cross-entropy). Then a DINO ceiling run: DINO ViT-S/16 with a DPT-Lite head, CE +
Lovász + auxiliary loss, AdamW with layer-wise learning-rate decay, warmup + cosine
schedule, augmentation and multi-scale + flip test-time augmentation.

## Results

Controlled comparison — Cityscapes val mIoU (%)

| Backbone | Pre-training | Linear probe | Full fine-tune |
|---|---|:---:|:---:|
| ResNet-50 | random init | — | 34.40 |
| ResNet-50 | ImageNet supervised | 43.27 | 53.62 |
| ResNet-50 | MoCo v3 | 43.61 | 54.69 |
| ViT-S/16  | DINO | 43.66 | 55.26 |

DINO fine-tuning — Cityscapes val mIoU (%)

| Setting | mIoU |
|---|:---:|
| EMA weights, full-res | 68.72 |
| EMA weights, full-res + multi-scale & flip TTA | 69.99 |

## Repository

- `cityscapes_eda.py` — exploratory data analysis
- `model_experiments.ipynb` — controlled backbone comparison
- `dino_finetune.ipynb` — DINO ViT-S/16 fine-tuning
