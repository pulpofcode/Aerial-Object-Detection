import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Aerial Object Classification",
    page_icon="🛸",
    layout="centered"
)

st.title("🛸 Aerial Object Classification")
st.write("Classify aerial images as **Bird** or **Drone** using Deep Learning.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 1)
    )
    model.load_state_dict(
        torch.load("mobilenet_transfer.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

uploaded_file = st.file_uploader(
    "Upload an aerial image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    threshold = 0.4  # security-focused
    label = "Drone 🚁" if prob >= threshold else "Bird 🐦"
    confidence = prob if prob >= threshold else (1 - prob)

    st.markdown("---")
    st.subheader("Prediction")
    st.write(f"**Class:** {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
