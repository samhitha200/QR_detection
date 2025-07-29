import streamlit as st
import numpy as np
import cv2
from PIL import Image
from joblib import load
from feature_extractor import extract_white_area_features
from io import BytesIO
import base64

# Load model
model = load("rf_white_features.pkl")

# Page config
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

st.markdown("""
    <style>
    /* Hide all scrollbars */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        overflow: hidden !important;
    }

    ::-webkit-scrollbar {
        width: 0px;
        background: transparent;
    }

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

    /* Adaptive Divider Line */
    .divider-line {
        height: 60vh;
        width: 2px;
        background-color: black; /* visible in light theme */
        margin: 0 auto;
        opacity: 0.8;
    }

    @media (prefers-color-scheme: dark) {
        .divider-line {
            background-color: white; /* visible in dark theme */
        }
    }

    .stButton>button {
        font-size: 20px !important;
        padding: 15px 24px !important;
        border: 2px solid #e74c3c !important;
        color: #e74c3c !important;
        background-color: transparent !important;
        border-radius: 8px !important;
    }

    .result-card {
        padding: 0.7rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.15);
        margin-top: 30px;
        border: 2px solid;
        width: 60%;
        margin-left: 160px;
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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="header-container">
        <h1>QR Code Authenticity Validator</h1>
        <p>Distinguish between Original vs Recaptured QR codes</p>
    </div>
""", unsafe_allow_html=True)

# Base64 encoder for image preview
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# 3 Column Layout: Left | Divider | Right
col_left, col_divider, col_right = st.columns([0.53, 0.02, 0.45])

image_pil = None

# --- LEFT PANEL: Upload and Display ---
with col_left:
    uploaded_file = st.file_uploader("📤 Upload a QR Code image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        resized = image_pil.copy()
        resized.thumbnail((300, 300))
        img_base64 = get_image_base64(resized)

        st.markdown(
           f"""
            <div style='text-align: center; padding: 10px;'>
                <img src='data:image/jpeg;base64,{img_base64}'
                    style='border-radius: 10px; max-width: 300px; height: auto;' />
            </div>
            """,
            unsafe_allow_html=True
        )

with col_divider:
    st.markdown("""
        <div style="
            height: 70vh;
            width: 2px;
            margin: 0 auto;
            background-color: #999;
            opacity: 0.8;
        "></div>
    """, unsafe_allow_html=True)

# --- RIGHT PANEL: Centered Button & Result ---
with col_right:
    if image_pil:
        # Use internal columns to center the button manually
        col_a, col_btn, col_b = st.columns([0.6, 0.5, 0.38])
        with col_btn:
            st.markdown("<div style='margin-top: 40px;'>", unsafe_allow_html=True)  # increase margin-top
            verify_clicked = st.button("🔍 Verify QR", key="verify_button")
            st.markdown("</div>", unsafe_allow_html=True)

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
                <div class='result-card {card_class}'>
                <div style='font-size: 2rem; text-transform: uppercase; letter-spacing: 1px;'>
                {label}
                </div>
                <div style='font-size: 1 rem; margin-top: 6px; font-weight: normal;'>
            Confidence: {confidence:.2f}%
        </div>
    </div>
""", unsafe_allow_html=True)

            else:
                st.warning("⚠️ Could not extract white area features.")
