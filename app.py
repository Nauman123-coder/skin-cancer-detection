# skin_cancer_app.py
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import cv2
import gradio as gr

# -----------------------------
# Load model
# -----------------------------
def load_model(model_path, device="cpu"):
    model = efficientnet_b0(num_classes=7)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

# -----------------------------
# Grad-CAM (FIXED + CLEANED)
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)  # ← fixes warning

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        logit = self.model(x)
        if class_idx is None:
            class_idx = logit.argmax(dim=1).item()

        self.model.zero_grad()
        score = logit[:, class_idx].sum()
        score.backward()

        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (gradients * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()   # ← THIS LINE FIXED IT!

# -----------------------------
# Prediction function — FINAL VERSION
# -----------------------------
def predict_and_visualize(img_pil):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    x = transform(img_pil).unsqueeze(0).to(device)
    x.requires_grad_()  # Needed for Grad-CAM

    # Prediction
    with torch.no_grad():
        logit = model(x)
        probs = torch.softmax(logit, dim=1).squeeze(0)
        pred_idx = probs.argmax().item()

    # Grad-CAM
    target_layer = model.features[-1][0]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(x, pred_idx)
    cam = cv2.resize(cam, (224, 224))

    # Heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    result_dict = {name: f"{p.item():.4f}" for name, p in zip(label_names, probs)}

    return result_dict, img_pil.resize((224, 224)), overlay

# -----------------------------
# Load model
# -----------------------------
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(r"E:\Skin Cancer Detection\Saved Model\skin_cancer_model.pth", device=device)
print("Model loaded successfully!")

# -----------------------------
# Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil", label="Upload skin lesion"),
    outputs=[
        gr.Label(num_top_classes=7, label="Predictions"),
        gr.Image(label="Original"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
    title="Skin Cancer Detection (HAM10000)",
    description="EfficientNet-B0 + Real-time Grad-CAM",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(share=True)



# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# from torchvision.models import efficientnet_b0
# from PIL import Image
# import numpy as np
# import cv2
# import pandas as pd 

# # Configuration
# # <<< CRITICAL: CHANGE THIS TO YOUR ACTUAL MODEL PATH! >>>
# MODEL_PATH = "Saved Model/skin_cancer_model.pth"
# LABEL_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# # -----------------------------
# # Load model (Cached for efficiency)
# # -----------------------------
# @st.cache_resource
# def load_model(model_path, num_classes=7, device="cpu"):
#     """Loads the PyTorch model from the specified path."""
#     try:
#         model = efficientnet_b0(num_classes=num_classes)
#         # Use map_location to ensure it loads correctly regardless of device (CPU or GPU)
#         state_dict = torch.load(model_path, map_location=device)
#         model.load_state_dict(state_dict)
#         model.eval()
#         model.to(device)
#         return model
#     except FileNotFoundError:
#         st.error(f"Error: Model file not found.")
#         return None
#     except Exception as e:
#         st.error(f"Error loading model.")
#         return None

# # -----------------------------
# # Grad-CAM Class (Intact)
# # -----------------------------
# class GradCAM:
#     """Class to compute Grad-CAM heatmap, using the logic from your original script."""
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         target_layer.register_forward_hook(self.save_activation)
#         target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def __call__(self, x, class_idx=None):
#         logit = self.model(x)
#         if class_idx is None:
#             class_idx = logit.argmax(dim=1).item()

#         self.model.zero_grad()
#         score = logit[:, class_idx].sum()
#         score.backward()

#         gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
#         cam = (gradients * self.activations).sum(dim=1).squeeze(0)
#         cam = torch.relu(cam)
#         cam = cam / (cam.max() + 1e-8)
#         return cam.detach().cpu().numpy()

# # -----------------------------
# # Prediction function
# # -----------------------------
# def predict_and_visualize(model, img_pil):
#     """
#     Performs inference, calculates Grad-CAM, and prepares outputs.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     x = transform(img_pil).unsqueeze(0).to(device)
#     x.requires_grad_() 

#     # Inference (Prediction)
#     with torch.no_grad():
#         logit = model(x)
#         probs = torch.softmax(logit, dim=1).squeeze(0)
#         pred_idx = probs.argmax().item()

#     # Grad-CAM calculation
#     target_layer = model.features[-1][0]
#     gradcam = GradCAM(model, target_layer)
#     cam = gradcam(x, pred_idx)
#     cam = cv2.resize(cam, (224, 224))

#     # Heatmap overlay visualization
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     img_np = np.array(img_pil.resize((224, 224)))
#     overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

#     result_dict = {name: probs[i].item() for i, name in enumerate(LABEL_NAMES)}
#     original_resized = img_pil.resize((224, 224))

#     return result_dict, original_resized, overlay

# # -----------------------------
# # Streamlit App Layout
# # -----------------------------

# st.set_page_config(
#     page_title="Skin Cancer Analyzer",
#     layout="wide",
# )

# # Main Title (The only required text)
# st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Skin Lesion Analyzer</h1>", unsafe_allow_html=True)

# # 1. Load Model
# model = load_model(MODEL_PATH, device="cpu")

# if model is None:
#     st.stop() 

# # 2. File Uploader (Minimalist)
# # Center the file uploader visually
# with st.container():
#     col_empty, col_up, col_empty2 = st.columns([1, 2, 1])
#     with col_up:
#         uploaded_file = st.file_uploader(
#             "Upload an image",
#             type=['jpg', 'jpeg', 'png'],
#             label_visibility="hidden"
#         )


# if uploaded_file is not None:
#     try:
#         img_pil = Image.open(uploaded_file).convert("RGB")
        
#         # Run prediction and visualization
#         results, original_resized, overlay = predict_and_visualize(model, img_pil)

#         # 3. Display Results using Streamlit columns
#         st.markdown("<hr style='border: 1px solid #ddd; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
#         col1, col2, col3 = st.columns([1, 1, 1])

#         with col1:
#             st.markdown("<p style='text-align: center; font-weight: bold;'>Original</p>", unsafe_allow_html=True)
#             # Changed use_column_width=True to use_container_width=True
#             st.image(original_resized, use_container_width=True)

#         with col2:
#             st.markdown("<p style='text-align: center; font-weight: bold;'>Grad-CAM</p>", unsafe_allow_html=True)
#             # Changed use_column_width=True to use_container_width=True
#             st.image(overlay, use_container_width=True)
        
#         with col3:
#             # Prepare results data
#             df_results = pd.DataFrame(
#                 list(results.items()), 
#                 columns=['Diagnosis', 'Probability']
#             ).sort_values(by='Probability', ascending=False)
            
#             df_results['Probability'] = df_results['Probability'].apply(lambda x: f"{x:.4f}")

#             top_diagnosis = df_results.iloc[0]['Diagnosis'].upper()
#             top_prob = df_results.iloc[0]['Probability']

#             st.markdown(f"<p style='text-align: center; font-weight: bold;'>Results</p>", unsafe_allow_html=True)

#             # Display top prediction clearly (Gradio-like result box)
#             st.markdown(
#                 f"""
#                 <div style='text-align: center; padding: 12px; background-color: #E3F2FD; border: 1px solid #BBDEFB; border-radius: 8px; margin-bottom: 15px;'>
#                     <span style='font-size: 1.3em; font-weight: bold; color: #1E88E5;'>{top_diagnosis}</span><br>
#                     <span style='font-size: 0.9em; color: #555;'>Confidence: {top_prob}</span>
#                 </div>
#                 """, unsafe_allow_html=True
#             )
            
#             # Display all probabilities (minimal table)
#             st.dataframe(
#                 df_results, 
#                 hide_index=True, 
#                 use_container_width=True,
#             )
            
#             st.balloons()
            
#     except Exception as e:
#         st.error("An unexpected error occurred during analysis.")