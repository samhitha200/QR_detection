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

# Minimal CSS
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
        margin-top: 1rem;
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

# State per image
if "results" not in st.session_state:
    st.session_state.results = {}

# Upload multiple files
uploaded_files = st.file_uploader("Upload one or more QR Code images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        img_id = f"img_{idx}"
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"Image {idx+1}")
            image_pil = Image.open(uploaded_file).convert("RGB")
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

        with col2:
            if st.button(f"🔍 Verify QR - Image {idx+1}"):
                image_np = np.array(image_pil)
                image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                features = extract_white_area_features(image_cv2)

                if features is not None:
                    prediction = model.predict([features])[0]
                    proba = model.predict_proba([features])[0]
                    label = "Original" if prediction == 0 else "Recaptured"
                    confidence = np.max(proba) * 100
                    st.session_state.results[img_id] = (label, confidence)
                else:
                    st.session_state.results[img_id] = ("Error", 0)

            # Show result card (if available)
            if img_id in st.session_state.results:
                label, conf = st.session_state.results[img_id]
                if label == "Error":
                    st.warning("⚠️ Could not extract white area features.")
                else:
                    card_class = "original" if label == "Original" else "recaptured"
                    st.markdown(f"""
                        <div class='result-card {card_class}'>
                            {label}<br/>
                            <span style='font-size: 0.9rem;'>Confidence: {conf:.2f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
