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
    .result-card {
        padding: 0.7rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.15);
        margin-top: 2rem;
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
    .divider-line {
        height: 100%;
        width: 2px;
        background-color: #999;
        margin: 0 auto;
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

# Helper
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# ---- Main layout
col_left, col_divider, col_right = st.columns([0.53, 0.01, 0.46])

# ---- LEFT PANEL
with col_left:
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

# ---- DIVIDER
with col_divider:
    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

# ---- RIGHT PANEL (aligned verify button)
with col_right:
    if uploaded_file:
        # Create inline block container for alignment
        st.markdown(
            """
            <div style="display: flex; justify-content: center; padding-top: 12px;">
                <button onclick="document.getElementById('verify-btn-hidden').click()" style="
                    font-size: 16px;
                    padding: 10px 28px;
                    border-radius: 6px;
                    background-color: #1f77b4;
                    color: white;
                    border: none;
                    cursor: pointer;
                ">🔍 Verify QR</button>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Hidden Streamlit button to handle logic
        verify_clicked = st.button("Verify QR", key="verify-btn-hidden")

        if verify_clicked:
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
                    <div class='result-card {card_class}' style='margin-top: 50px;'>
                        {label}<br/>
                        <span style='font-size: 0.95rem;'>Confidence: {confidence:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Could not extract white area features.")
