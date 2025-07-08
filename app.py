import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image
from feature_extractor import extract_white_area_features

# Load the model
model = load("rf_white_features.pkl")

# Page setup
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stFileUploader {
        padding: 0.3rem 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stFileUploader > div:first-child {
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown("<h1 style='text-align: center;'>QR Code Authenticity Validator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Distinguish between Original vs Recaptured QR codes</p>", unsafe_allow_html=True)

# File uploader placed just below header
uploaded_file = st.file_uploader("Upload a QR Code image (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

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

            # Layout: image and result
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, channels="BGR", caption="Uploaded QR Code", width=350)  # 👈 Increased size

            with col2:
                st.markdown("### Prediction")
                st.markdown(f"**Result:** `{label}`")
                st.markdown(f"**Confidence:** `{confidence:.2f}%`")
        else:
            st.warning("Not enough white area detected in the image to extract features.")
    else:
        st.error("Could not decode image. Please upload a valid file.")
