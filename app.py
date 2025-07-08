import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image
from feature_extractor import extract_white_area_features
import base64
from io import BytesIO

# Load model
model = load("rf_white_features.pkl")

# Page config
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

# Custom style to reduce padding and heading size
st.markdown("""
    <style>
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h1 {
        font-size: 1.75rem;
        margin-bottom: 0.25rem;
    }
    p {
        font-size: 0.9rem;
        margin-top: 0.1rem;
        margin-bottom: 0.4rem;
    }
    .stFileUploader {
        padding: 0.2rem 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>QR Code Authenticity Validator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Distinguish between Original vs Recaptured QR codes</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a QR Code image (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

# Helper to display image with fixed size
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image is not None:
        features = extract_white_area_features(image)

        if features is not None:
            prediction = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            label = "Original" if prediction == 0 else "Recaptured"
            confidence = np.max(proba) * 100

            # Display layout
            col1, col2 = st.columns([1, 1])
            with col1:
                resized = image_pil.copy()
                resized.thumbnail((400, 400))  # Resize for display
                img_base64 = get_image_base64(resized)
                st.markdown(
                    f"<div style='display: flex; justify-content: center;'><img src='data:image/jpeg;base64,{img_base64}' style='border-radius: 10px;'/></div>",
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown("### Prediction")
                st.markdown(f"**Result:** `{label}`")
                st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        else:
            st.warning("Not enough white area detected in the image to extract features.")
    else:
        st.error("Could not decode image. Please upload a valid file.")
