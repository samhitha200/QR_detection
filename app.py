import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image
from feature_extractor import extract_white_area_features

# Load the model
model = load("rf_white_features.pkl")

# Configure Streamlit page
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

# --- Custom CSS to reduce top padding ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 {
            margin-bottom: 0.2rem;
        }
        .stCaption {
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and caption
st.title("QR Code Authenticity Validator")
st.caption("Distinguish between Original vs Recaptured QR codes")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a QR Code image (.jpg/.jpeg/.png)",
    type=["jpg", "jpeg", "png"]
)

# Process if file is uploaded
if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    features = extract_white_area_features(image)

    if features is not None:
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        label = "Original" if prediction == 0 else "Recaptured"
        confidence = np.max(proba) * 100

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, channels="BGR", caption="Uploaded QR Code", use_container_width=True)

        with col2:
            st.markdown("### Prediction")
            st.markdown(f"**Result:** `{label}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")
    else:
        st.warning("Not enough white area detected in the image to extract features.")
