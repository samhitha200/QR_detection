import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image
from feature_extractor import extract_white_area_features
from io import BytesIO
import base64

# Load the trained model
model = load("rf_white_features.pkl")

# Configure Streamlit page
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

# Header styling at top-left corner
st.markdown("""
    <style>
    .custom-header {
        position: absolute;
        top: 10px;
        left: 20px;
        font-size: 1.4rem;
        font-weight: 600;
        color: #333333;
        z-index: 1000;
    }
    .custom-subtitle {
        position: absolute;
        top: 40px;
        left: 20px;
        font-size: 0.9rem;
        color: #666666;
        z-index: 1000;
    }
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
    <div class="custom-header">QR Code Authenticity Validator</div>
    <div class="custom-subtitle">Distinguish between Original vs Recaptured QR codes</div>
""", unsafe_allow_html=True)

# Helper to convert image to base64 for display
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

# Layout: Two columns (left for upload + preview, right for processing + result)
left_col, right_col = st.columns([1, 1.2])

# 📤 Left Column — Upload and preview
with left_col:
    uploaded_file = st.file_uploader("Upload QR Code Image", type=["jpg", "jpeg", "png"])
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

#  Right Column — Button + Result
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

    # Show result
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
