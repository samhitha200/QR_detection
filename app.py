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

# CSS styling
st.markdown("""
    <style>
    .header-container {
        padding-top: 0.2rem;
        margin-bottom: -1rem;
    }
    .header-container h1 {
        font-size: 1.5rem;
        color: var(--text-color);  
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .header-container p {
        font-size: 0.9rem;
        color: var(--text-color);  
        text-align: center;
        margin-top: 0;
    }
    .result-card {
        padding: 0.7rem;
        border-radius: 12px;
        color: var(--text-color); 
        font-weight: bold;
        text-align: center;
        font-size: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.15);
        margin-top: 2.5rem;  /* More space added here */
        border: 2px solid;
        width: 60%;
        margin-left: auto;
        margin-right: auto;
    }
    .original {
        background-color: #2e7d32;
        border-color: #1b5e20;
    }
    .recaptured {
        background-color: #ef6c00;
        border-color: #bf360c;
    }
    .verify-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    .separator {
        border-left: 2px solid #555;
        height: 100%;
        margin: 0 1rem;
    }
    button[kind="secondary"] {
        font-size: 1.2rem !important;
        padding: 0.8rem 2rem !important;
        border-radius: 8px !important;
        border: 2px solid #ef5350 !important;
        background-color: #1e1e1e !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="header-container">
        <h1>QR Code Authenticity Validator</h1>
        <p>Distinguish between Original vs Recaptured QR codes</p>
    </div>
""", unsafe_allow_html=True)

# Helper to encode image to base64
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# Layout with separator
left_col, spacer, right_col = st.columns([0.45, 0.02, 0.53])

# Left: Upload and preview
with left_col:
    uploaded_file = st.file_uploader("Upload a QR Code image", type=["jpg", "jpeg", "png"])
    image_pil = None
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        resized = image_pil.copy()
        resized.thumbnail((400, 400))
        img_base64 = get_image_base64(resized)
        st.markdown(
            f"<div style='text-align: center;'><img src='data:image/jpeg;base64,{img_base64}' style='border-radius: 10px;'/></div>",
            unsafe_allow_html=True
        )

# Vertical line separator
with spacer:
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

# Right: Button and Results
with right_col:
    st.markdown('<div class="verify-container">', unsafe_allow_html=True)
    verify_clicked = st.button("🔍 Verify QR", key="verify_button_centered")
    st.markdown('</div>', unsafe_allow_html=True)

    if verify_clicked:
        if image_pil is not None:
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
                    <div class='result-card {card_class}'>
                        {label}<br/>
                        <span style='font-size: 0.85rem;'>Confidence: {confidence:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Could not extract white area features.")
        else:
            st.warning("⚠️ Please upload a QR code image first.")
