import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image
from feature_extractor import extract_white_area_features
from io import BytesIO
import base64

# Load model
model = load("rf_white_features.pkl")

# Page config
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

# Background image CSS
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1612831455542-d9b56b3efcb0?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 2rem;
        backdrop-filter: blur(4px);
    }
    h1, p {
        text-align: center;
    }
    .result-card {
        padding: 1rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        margin-top: 1rem;
    }
    .original { background-color: #2e7d32; }
    .recaptured { background-color: #ef6c00; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>QR Code Authenticity Validator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Distinguish between Original vs Recaptured QR codes</p>", unsafe_allow_html=True)

# Helper to get base64 from image
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# File upload
uploaded_files = st.file_uploader("Upload one or more QR Code images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Process each file
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.subheader(f"Image {idx+1}")
            image_pil = Image.open(uploaded_file).convert("RGB")
            resized = image_pil.copy()
            resized.thumbnail((400, 400))
            img_base64 = get_image_base64(resized)
            st.markdown(
                f"<div style='text-align: center;'><img src='data:image/jpeg;base64,{img_base64}' style='border-radius: 10px;'/></div>",
                unsafe_allow_html=True
            )

        with col2:
            verify_button = st.button(f"🔍 Verify QR - Image {idx+1}")

            if verify_button:
                image_np = np.array(image_pil)
                image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                features = extract_white_area_features(image_cv2)

                if features is not None:
                    prediction = model.predict([features])[0]
                    proba = model.predict_proba([features])[0]
                    label = "Original" if prediction == 0 else "Recaptured"
                    confidence = np.max(proba) * 100

                    card_class = "original" if label == "Original" else "recaptured"
                    st.markdown(
    f"""
    <div style='text-align: center; padding: 10px; background-color: #f0f0f0; border: 1px solid #888; border-radius: 12px; display: inline-block;'>
        <img src='data:image/jpeg;base64,{img_base64}' style='max-width: 100%; border-radius: 8px;'/>
    </div>
    """,
    unsafe_allow_html=True
)

                else:
                    st.warning("⚠️ Could not extract white area features.")
