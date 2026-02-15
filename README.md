# MARCo: Masked Autoencoding with Radar–Optical Contrast for Multimodal Remote Sensing

This repository contains the official implementation of **MARCo**, a cross-modal masked autoencoder for multimodal remote sensing, jointly learning from **Sentinel-1 (SAR)** and **Sentinel-2 (optical)** imagery.

---

## Dataset

This work uses the **IEEE GRSS Data Fusion Contest 2020 (DFC 2020)** benchmark dataset.

Used the **preprocessed DFC 2020 subset** released by the CROMA authors to ensure standardized splits and reproducible evaluation:

🔗 https://huggingface.co/datasets/antofuller/CROMA_benchmarks  

Only the **DFC 2020 subset** from the CROMA benchmark collection is used in this repository.

If you use this dataset, please cite:

- The original **DFC 2020** paper  
- The **CROMA benchmark release**

---

## Reproducibility

To facilitate reproducibility, pretrained model checkpoints are available here:

🔗 https://drive.google.com/drive/folders/1agxUrlOoDHzF7dKRF0xQ3oVNEnIrLRZf?usp=sharing  

These checkpoints correspond to the configurations in MARCo_v4.
