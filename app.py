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
    h1, p { text-align: center; }
    .result-card {
        padding: 1rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    .original { background-color: #2e7d32; }
    .recaptured { background-color: #ef6c00; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>QR Code Authenticity Validator</h1>", unsafe_allow_html=True)
st.markdown("<p>Distinguish between Original vs Recaptured QR codes</p>", unsafe_allow_html=True)

# Helper
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# Session state
if "image_pil" not in st.session_state:
    st.session_state.image_pil = None
if "result" not in st.session_state:
    st.session_state.result = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None

# Two-pane layout
left_col, right_col = st.columns([1, 1.2])

# 🔲 Left Panel — Upload and Image Preview
with left_col:
    uploaded_file = st.file_uploader("📤 Upload QR Code Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.session_state.image_pil = image_pil

        resized = image_pil.copy()
        resized.thumbnail((400, 400))
        img_base64 = get_image_base64(resized)

        st.markdown(
            f"""
            <div style='text-align: center; padding: 10px; background-color: #f0f0f0; border: 1px solid #888; border-radius: 12px; display: inline-block;'>
                <img src='data:image/jpeg;base64,{img_base64}' style='max-width: 100%; border-radius: 8px;'/>
            </div>
            """,
            unsafe_allow_html=True
        )

# 🔳 Right Panel — Button + Result Card
with right_col:
    st.write("### ")
    st.write("### ")
    if st.button("🔍 Verify QR"):
        if st.session_state.image_pil:
            image_np = np.array(st.session_state.image_pil)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            features = extract_white_area_features(image_cv2)

            if features is not None:
                prediction = model.predict([features])[0]
                proba = model.predict_proba([features])[0]
                label = "Original" if prediction == 0 else "Recaptured"
                confidence = np.max(proba) * 100

                st.session_state.result = label
                st.session_state.confidence = confidence
            else:
                st.warning("⚠️ Could not extract white area features.")

    if st.session_state.result:
        label = st.session_state.result
        conf = st.session_state.confidence
        card_class = "original" if label == "Original" else "recaptured"
        st.markdown(f"""
            <div class='result-card {card_class}'>
                {label}<br/>
                <span style='font-size: 0.9rem;'>Confidence: {conf:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
