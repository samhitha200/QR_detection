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
        width: 1px;
        background-color: #888;
        margin: auto;
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
    .right-panel-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        height: 100%;
        padding-top: 2rem;
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

# Layout with a vertical divider (60 | 1 | 39)
left_col, divider_col, right_col = st.columns([0.6, 0.01, 0.39])

# Left Panel: Upload & Display
with left_col:
    uploaded_file = st.file_uploader("📤 Upload a QR Code image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        resized = image_pil.copy()
        resized.thumbnail((500, 500))
        img_base64 = get_image_base64(resized)
        st.markdown(
            f"<div style='text-align: center;'><img src='data:image/jpeg;base64,{img_base64}' "
            f"style='border-radius: 10px; max-width: 100%; height: auto;'/></div>",
            unsafe_allow_html=True
        )

# Divider line
with divider_col:
    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

# Right Panel with visual separator and centered content
with right_col:
    if uploaded_file:
        st.markdown("""
            <div style="
                border-left: 1px solid #666;
                padding-left: 30px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
            ">
        """, unsafe_allow_html=True)

        verify_button = st.button("🔍 Verify QR")

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
                st.markdown(f"""
                    <div class='result-card {card_class}'>
                        {label}<br/>
                        <span style='font-size: 0.85rem;'>Confidence: {confidence:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Could not extract white area features.")

        st.markdown("</div>", unsafe_allow_html=True)
