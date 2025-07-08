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

# Centered Header (keep as is)
st.markdown("""
    <style>
    .header-container {
        padding-top: 0.2rem;
        margin-bottom: -1rem;
    }
    .header-container h1 {
        font-size: 1.5rem;
        color: white;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .header-container p {
        font-size: 0.9rem;
        color: white;
        text-align: center;
        margin-top: 0;
    }
    </style>
    <div class="header-container">
        <h1>QR Code Authenticity Validator</h1>
        <p>Distinguish between Original vs Recaptured QR codes</p>
    </div>
""", unsafe_allow_html=True)

# Result card styles
st.markdown("""
    <style>
    .result-card {
        padding: 1.2rem;
        border-radius: 14px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        margin-top: 1rem;
        border: 3px solid;
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

# Helper to get base64 from image
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# Two-pane layout
left_col, right_col = st.columns([1, 1.2])

# Left: Upload + display
with left_col:
    uploaded_file = st.file_uploader("Upload a QR Code image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)

# Right: Verify button + result
with right_col:
    if uploaded_file:
        verify_button = st.button("🔍 Verify QR")

        if verify_button:
            image_pil = Image.open(uploaded_file).convert("RGB")
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
                        <span style='font-size: 0.95rem;'>Confidence: {confidence:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Could not extract white area features.")
