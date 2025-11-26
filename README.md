# Skin Cancer Detection Using Deep Learning  
**Early Detection Saves Lives**

### Project Overview
This repository contains a complete, ready-to-use skin cancer classification system built with **EfficientNet-B0** and the **HAM10000** dataset. The model classifies dermatoscopic images into **7 different skin lesion types** and includes **real-time Grad-CAM visualization** to show doctors exactly which parts of the lesion the AI focused on.

The project includes:
- A Jupyter notebook for inference & Grad-CAM testing (`skin_cancer_detection_lab.ipynb`)
- A beautiful, shareable web demo using **Gradio** (`app.py`)
- A clean, production-ready inference pipeline

---

### Why AI for Skin Cancer Detection?

- Skin cancer is the **most common cancer** globally.
- **Melanoma**, though only ~1% of cases, causes the majority of skin cancer deaths.
- 5-year survival rate: **>99%** if detected early → **<25%** if metastasized.
- Dermatologist-level visual assessment is time-consuming and not available everywhere.
- AI can act as a **fast, reliable screening tool** to flag suspicious lesions for biopsy, reducing diagnostic delay and improving outcomes.

This model helps bridge that gap by providing **instant, explainable predictions**.

---

### Dataset: HAM10000  
("Human Against Machine with 10,000 training images")

| Property              | Details                              |
|-----------------------|--------------------------------------|
| Total Images          | 10,015                               |
| Classes               | 7                                    |
| Resolution            | Variable (mostly 600×450)            |
| Source                | ISIC Archive (Kaggle)                |
| Classes               | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` |

**Class Distribution** (highly imbalanced – `nv` dominates ~67%)  
Proper handling of class imbalance was applied during training.

---

### Model Architecture

- **Base Model**: `EfficientNet-B0` (pre-trained on ImageNet)
- **Input Size**: 224 × 224 × 3
- **Final Layer**: Modified to output **7 classes**
- **Framework**: PyTorch
- **Saved Model**: `skin_cancer_model.pth` (full model + weights)
- **Key Feature**: **Grad-CAM** integration for interpretability

The model achieves strong performance on the HAM10000 benchmark while remaining lightweight enough to run on standard laptops.

---

### How It Helps Doctors

| Feature                  | Benefit to Clinicians                              |
|--------------------------|-----------------------------------------------------|
| Instant 7-class prediction | Rapid risk stratification                         |
| Grad-CAM heatmaps         | Shows exactly where the model "looked"            |
| Confidence scores         | Helps decide urgency of biopsy                    |
| Web interface (Gradio)    | No installation needed – shareable link           |
| Explainable AI            | Builds trust and supports clinical decision-making |

Doctors can use this as a **second-opinion tool** during screening, especially in primary care or underserved regions.

---

### Quick Start

```bash
# 1. Clone repo
git clone https://github.com/Nauman123-coder/skin-cancer-detection.git
cd skin-cancer-detection

# 2. Install dependencies
pip install torch torchvision gradio opencv-python pillow numpy

# 3. Run the web demo
python app.py
