# 🧬 Biomedical Image Segmentation with Flexible UNet Variants

Real-world biomedical imaging tasks often face challenges of dataset scarcity, imbalance, and variability across modalities. This repository provides an **end-to-end biomedical image segmentation pipeline** built on UNet and its latest variants. The aim is to show that while **UNets are powerful segmentation models**, combining them with flexible loss functions and augmentation strategies improves generalization across diverse medical datasets.

---

## 📌 Features

- Support for **multiple UNet variants**: UNet, Attention UNet, UNet++, DC-UNet, Half-UNet  
- Flexible **loss functions**: Cross-Entropy (CE), Dice Loss, Focal Cross-Entropy, Tversky Loss, Boundary Dice Loss  
- Modular **augmentation pipelines** designed for medical images  
- Configurable training pipeline supporting multiple datasets  
- Benchmarking across ISIC, ACDC, and microscopy datasets  

---

---

## ⚙️ Installation

Clone the repository and install dependencies:  
Run `git clone <repo_url>` then `cd biomedical-image-segmentation`  
Run `pip install -r requirements.txt`  

---

## 📦 Datasets

This project was trained and validated on three datasets:  
- ISIC (Skin Lesion Segmentation) – public dataset, not included here  
- ACDC (Cardiac MRI Segmentation) – public dataset, not included here  
- Mouse Embryonic Stem Cell Microscopy – internal dataset, not included here  

Datasets should be structured as:  

datasets/  
├── ISIC/  
│   ├── train/  
│   ├── val/  
│   └── test/  
├── ACDC/  
│   ├── train/  
│   ├── val/  
│   └── test/  
└── StemCell/  
    ├── train/  
    ├── val/  
    └── test/  

 Note: Due to size and licensing, datasets are not included. Please download them manually from their official sources or substitute with your own data. You could also use your custom medical image dataset and train it using this pipeline, the model is built to generalise across medical imaging distributions.

---

##  Training

Train models with configurable architecture and loss function. Example:  
Run `python train.py --model attention_unet --loss dice --dataset ISIC`  

Options:  
- `--model`: unet, attention_unet, unetpp, dcunet, halfunet etc
- `--loss`: ce, dice, focalce, tversky, boundary_dice etc
- `--dataset`: ISIC, ACDC, StemCell  

---

## 🔍 Validation

Run `python evaluate.py --model unetpp --dataset ACDC`  
This outputs F1 and Dice score 

---

## Baseline Results on UNET model training

| Dataset  | Model           | Dice | F1  |  
|----------|-----------------|------|------|  
| ISIC     |  UNet- CE + Dice loss | 0.87 | 0.77 |  
| ACDC     | UNet- CE + Dice loss       | 0.98 | 0.92 |  
| StemCell | UNet - CE + Tversky        | 0.88 | 0.74 |  

---

##  Sample Segmentation results from training UNET on these three datasets with tailored loss functions and augmentations

Below are sample input images and their predicted segmentations across the three datasets.

<table>
  <tr>
    <th>ISIC Input</th>
    <th>ISIC Prediction</th>
  </tr>
  <tr>
    <td><img src="results/isic_input.png" width="250"></td>
    <td><img src="results/isic_pred.png" width="250"></td>
  </tr>
  <tr>
    <th>ACDC Input</th>
    <th>ACDC Prediction</th>
  </tr>
  <tr>
    <td><img src="results/acdc_input.png" width="250"></td>
    <td><img src="results/acdc_pred.png" width="250"></td>
  </tr>
  <tr>
    <th>StemCell Input</th>
    <th>StemCell Prediction</th>
  </tr>
  <tr>
    <td><img src="results/stemcell_input.png" width="250"></td>
    <td><img src="results/stemcell_pred.png" width="250"></td>
  </tr>
</table>


---



---

## 📜 Cite this Work

If you use or build upon this work, please cite:  

`@misc{ajith2025unetsegmentation,  
  author       = {Varun Ajith},  
  title        = {Biomedical Image Segmentation with Flexible UNet Variants},  
  year         = {2025},  
  url          = {https://github.com/<username>/Generalizable-Biomedical-Image-Segmentation},  
  note         = {UNet variants with flexible loss functions and augmentation pipelines for medical imaging}  
}`  

