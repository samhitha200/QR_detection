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
        padding-top: 0.2rem;
        margin-bottom: 1rem;
    }
    .header-container h1 {
        font-size: 1.5rem;
        color: var(--text-color);  
        text-align: center;
    }
    .header-container p {
        font-size: 0.9rem;
        color: var(--text-color);  
        text-align: center;
    }
    .result-card {
        padding: 1rem;
        border-radius: 12px;
        color: var(--text-color); 
        font-weight: bold;
        text-align: center;
        font-size: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.15);
        margin-top: 3rem;
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
    .separator {
        border-left: 2px solid #666;
        height: 500px;
        margin: auto;
    }
    .center-verify {
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        font-size: 18px !important;
        padding: 0.75rem 2rem !important;
        border: 2px solid #ef5350 !important;
        border-radius: 8px !important;
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

# Helper: Image to base64
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# Layout
left_col, mid_col, right_col = st.columns([0.59, 0.02, 0.44])

# Upload and preview (Left)
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

# Middle Column - Separator
with mid_col:
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

# Right Column - Centered Verify Button and Result
with right_col:
    st.markdown("<div class='center-right-panel'>", unsafe_allow_html=True)
    verify_clicked = st.button("🔍 Verify QR", key="verify_button")
    st.markdown("</div>", unsafe_allow_html=True)

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
