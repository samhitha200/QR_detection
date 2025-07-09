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

st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

# CSS Styling
st.markdown("""
    <style>
    .header-container {
        padding-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .header-container h1 {
        font-size: 1.5rem;
        text-align: center;
    }
    .header-container p {
        font-size: 0.9rem;
        text-align: center;
    }
    .divider-line {
        height: 100%;
        width: 2px;
        background-color: #888;
        margin: 0 auto;
    }
    .result-card {
        padding: 0.7rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.15);
        margin-top: 1.5rem;
        border: 2px solid;
        width: 100%;
    }
    .original {
        background-color: #2e7d32;
        border-color: #1b5e20;
    }
    .recaptured {
        background-color: #ef6c00;
        border-color: #bf360c;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1>QR Code Authenticity Validator</h1>
        <p>Distinguish between Original vs Recaptured QR codes</p>
    </div>
""", unsafe_allow_html=True)

# Image to base64 utility
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# Layout: Left (Upload) | Divider | Right (Verify & Result)
col_left, col_divider, col_right = st.columns([0.53, 0.02, 0.45])

# LEFT PANEL
with col_left:
    uploaded_file = st.file_uploader("📤 Upload a QR Code image", type=["jpg", "jpeg", "png"])
    image_pil = None
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        resized = image_pil.copy()
        resized.thumbnail((500, 500))
        img_b64 = get_image_base64(resized)
        st.markdown(f"<div style='text-align: center;'><img src='data:image/jpeg;base64,{img_b64}' style='max-width: 100%; border-radius: 8px;'/></div>", unsafe_allow_html=True)

# DIVIDER
with col_divider:
    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

# RIGHT PANEL
with col_right:
    if uploaded_file:
        verify_clicked = st.button("🔍 Verify QR", key="verify_button")
    else:
        st.info("⬆️ Upload a QR Code image to enable the Verify button.")
        verify_clicked = False

    if verify_clicked and image_pil is not None:
        image_np = np.array(image_pil)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        features = extract_white_area_features(image_cv2)

        if features is not None:
            prediction = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            label = "Original" if prediction == 0 else "Recaptured"
            confidence = np.max(proba) * 100
            card_class = "original" if label == "Original" else "recaptured"

            st.markdown(f"""
                <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; margin-top: 30px;'>
                    <button style="
                        font-size: 18px;
                        padding: 10px 24px;
                        border: 2px solid #e74c3c;
                        color: #e74c3c;
                        background-color: transparent;
                        border-radius: 8px;
                        cursor: not-allowed;
                        margin-bottom: 1.2rem;
                    " disabled>🔍 Verify QR</button>

                    <div class='result-card {card_class}'>
                        {label}<br/>
                        <span style='font-size: 0.95rem;'>Confidence: {confidence:.2f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Could not extract white area features.")
    elif verify_clicked:
        st.warning("⚠️ Please upload an image first.")
