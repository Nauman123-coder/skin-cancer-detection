---
title: Skin Cancer Detection with Grad-CAM
emoji: 🩺  # ← Fixed: real emoji, not text
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Skin Cancer Detection with Real-time Grad-CAM

**EfficientNet-B0** trained from scratch on the **HAM10000** dataset  
Detects **7 types of skin lesions** including **Melanoma** with **live attention heatmaps** (Grad-CAM)

### Classes Detected
| Code  | Full Name                                | Description                                      | Risk Level     |
|------|-------------------------------------------|----------------------------------------------------|----------------|
| mel  | Melanoma                                  | Most dangerous form of skin cancer                 | HIGH           |
| bcc  | Basal Cell Carcinoma                      | Most common skin cancer, rarely spreads           | Moderate       |
| akiec| Actinic Keratoses / Bowen’s disease      | Pre-cancerous, can progress to SCC                 | Pre-cancerous  |
| nv   | Melanocytic Nevi (moles)                  | Usually benign, but monitor changes                | Low            |
| bkl  | Benign Keratosis                          | Non-cancerous growths (seborrheic keratosis etc.)  | Benign         |
| df   | Dermatofibroma                            | Benign fibrous nodule                              | Benign         |
| vasc | Vascular Lesions                          | Benign blood vessel growths                        | Benign         |

### Features
- Real-time **Grad-CAM** visualization → see exactly where the model is looking  
- Fast inference (works on CPU & GPU)  
- No installation needed — just upload an image  
- Trained on 10,015 dermatoscopic images (HAM10000)

### How to Use
1. Upload a clear dermatoscopic (or high-quality close-up) image of a skin lesion  
2. Wait ~2 seconds  
3. Get instant prediction + heatmap showing the region of interest

**Important**: This is an AI research tool — **not a medical diagnosis**. Always consult a dermatologist.

### Model Details
- Architecture: EfficientNet-B0 (PyTorch + torchvision)  
- Input size: 224×224  
- Classes: 7  
- Trained for 10+ epochs with data augmentation  
- Uses ImageNet normalization