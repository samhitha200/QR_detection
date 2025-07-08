import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image
from io import BytesIO
import base64
from feature_extractor import extract_white_area_features

# Load the model
model = load("rf_white_features.pkl")

# Page setup
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

# Custom CSS
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

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>QR Code Authenticity Validator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Distinguish between Original vs Recaptured QR codes</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a QR Code image (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

# Function to display resized, centered image
def display_fixed_size_image(image_cv2):
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    max_size = (400, 400)  # Resize to fit within 400x400 while keeping aspect ratio
    pil_img.thumbnail(max_size, Image.ANTIALIAS)

    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; min-height: 420px;">
            <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 10px; border: 1px solid #444;" />
        </div>
    """, unsafe_allow_html=True)

# If image is uploaded
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

            # Display results
            col1, col2 = st.columns([1, 1])
            with col1:
                display_fixed_size_image(image)
            with col2:
                st.markdown("### Prediction")
                st.markdown(f"**Result:** `{label}`")
                st.markdown(f"**Confidence:** `{confidence:.2f}%`")
        else:
            st.warning("Not enough white area detected in the image to extract features.")
    else:
        st.error("Could not decode image. Please upload a valid file.")
