<h1 align="center">Self-Supervised Learning for Autonomous Driving</h1>

<p align="center">
  <em>Comparing self-supervised pre-training strategies for semantic segmentation on Cityscapes,<br/>
  then pushing the winning backbone to a practical mIoU ceiling.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Backbone-DINO%20ViT--S%2F16-1f6feb" alt="DINO ViT-S/16">
  <img src="https://img.shields.io/badge/Dataset-Cityscapes-1f6feb" alt="Cityscapes">
</p>

<p align="center">
  <a href="./Master_Dissertation_Daniil_Deltsov.pdf">Thesis</a>
  &nbsp;·&nbsp;
  <a href="./model_experiments.ipynb">Controlled comparison</a>
  &nbsp;·&nbsp;
  <a href="./dino_finetune.ipynb">DINO fine-tuning</a>
  &nbsp;·&nbsp;
  <a href="./dashcam_demo.ipynb">Dashcam demo</a>
  &nbsp;·&nbsp;
  <a href="./presentation/Deltsov_D_A_SSL.pptx">Slides</a>
</p>

---

<p align="center">
  <img src="assets/demo.gif" alt="Dashcam segmentation demo" width="820"/>
</p>

<p align="center">
  <b>69.99</b> mIoU &nbsp;·&nbsp; <b>Cityscapes val</b> &nbsp;·&nbsp; <b>DINO ViT-S/16 + DPT-Lite</b>
</p>

---

## Question

Do self-supervised pre-training strategies produce features that transfer better to urban-scene semantic segmentation than supervised baselines — and which method wins?

## Method

Two experiments answering two different questions.

|  | **Controlled comparison** | **DINO ceiling** |
|---|---|---|
| **Goal** | isolate pre-training as the only varying factor | push the winning backbone to its practical mIoU ceiling |
| **Backbones** | random · ImageNet supervised · MoCo v3 · DINO | DINO ViT-S/16 |
| **Modes** | linear probe + full fine-tune (7 cells) | full fine-tune |
| **Head** | FCN, 2× 3×3 conv | DPT-Lite multi-block fusion + auxiliary supervision |
| **Loss** | cross-entropy | CE + 0.5·Lovász + 0.4·aux |
| **Optim** | Adam, lr 2e-4, 20 ep | AdamW, 29-group LLRD (λ=0.65), warmup + cosine, 50 ep |
| **Aug / TTA** | none | scale-and-crop, flip, color-jitter · multi-scale + h-flip TTA |
| **Resolution** | 512 × 1024 | 768² crops · 1024 × 2048 eval |
| **Precision** | fp32 | bf16 mixed, EMA decay 0.999 |

## Results

**Controlled comparison** — Cityscapes val mIoU (%)

| Backbone | Pre-training | Linear probe | Full fine-tune |
|---|---|:---:|:---:|
| ResNet-50 | random init | — | 34.40 |
| ResNet-50 | ImageNet supervised | 43.27 | 53.62 |
| ResNet-50 | MoCo v3 | 43.61 | 54.69 |
| ViT-S/16  | DINO | 43.66 | **55.26** |

**DINO fine-tuning** — Cityscapes val mIoU (%)

| Setting | mIoU |
|---|:---:|
| EMA weights, full-res | 68.72 |
| EMA weights, full-res + multi-scale & flip TTA | **69.99** |

## Repository

```
.
├── model_experiments.ipynb     7-model controlled comparison
├── dino_finetune.ipynb         DINO ViT-S/16 fine-tuning experiment
├── dashcam_demo.ipynb          qualitative dashcam inference
├── cityscapes_eda.py           exploratory data analysis
├── eda_outputs/                EDA figures
└── assets/                     demo.gif
```

## Acknowledgements

Cityscapes (Cordts et al., 2016) for the dataset and benchmark. DINO (Caron et al., 2021) and MoCo v3 (Chen et al., 2021) for the pre-trained checkpoints used in the comparison. SLidR (Sautier et al., 2022) and S3PT (Wozniak et al., 2025) for grounding the evaluation protocol.