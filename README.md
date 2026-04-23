# Glacial Anomaly Detection 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yiymQ8BmM8QddM-PpOWjs3HCSyuX50Nq?usp=sharing)

## Overview
This repository contains a complete data science pipeline designed to detect and analyze glacial anomalies using semantic segmentation. The project focuses on processing multi-band satellite imagery to identify irregular patterns, demonstrating a practical application of deep learning to climate and geographic datasets. 

## Dataset
* **Source:** Geospatial RGB TIFF satellite images paired with binary ground-truth masks.
* **Description:** High-resolution satellite imagery used to extract fine-grained details of glacier anomalies. The provision of binary masks eliminated the need for complex radiometric preprocessing or manual raster-vector masking.

## Methodology & Tech Stack
The analysis and model training were conducted entirely within Google Colab using the PyTorch framework.
* **Data Preprocessing:** Images were resized into patches (e.g., 64x64) and processed using a positive-pixel filtering method with a 0.5 threshold to combat severe class imbalance. Standard data augmentations were applied.
* **Modeling:** Multiple semantic segmentation architectures were evaluated (U-Net, ENet, AGC-Net). The most successful approach utilized **Swin-UNet**, which combines a transformer-based encoder using 7x7 shifted-window attention with a U-Net style decoder.
* **Optimization:** Models were trained using the Adam optimizer with a combined Binary Cross-Entropy (BCE) and Dice Loss function.

## Key Results
* **Architecture Superiority:** Swin-UNet emerged as the best-performing model, achieving approximately 96% accuracy and 97.13% precision on a clean dataset.
* **Imbalance Handling:** The project successfully navigated severe class imbalances (where background regions dominate) by implementing targeted filtering and data augmentation.
* **Conclusion:** The findings highlight the significant advantages of transformer-based encoders for remote sensing segmentation tasks.
