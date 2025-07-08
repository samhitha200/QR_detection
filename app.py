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

# ----- CSS Styling -----
st.markdown("""
    <style>
    .header-container {
        padding-top: 0.2rem;
        margin-bottom: -1rem;
        text-align: center;
    }
    .header-container h1 {
        font-size: 1.5rem;
        color: var(--text-color);  
        margin-bottom: 0.2rem;
    }
    .header-container p {
        font-size: 0.9rem;
        color: var(--text-color);  
        margin-top: 0;
    }

    .right-align-block, .left-align-block {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        padding-top: 60px;
    }

    .stButton > button {
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        border-radius: 12px;
        background-color: #1e1e1e;
        color: white;
        border: 2px solid #ff4b4b;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #ff4b4b;
        color: white;
    }

    .result-card {
        margin-top: 2rem;
        padding: 1rem;
        font-size: 1.05rem;
        font-weight: bold;
        border-radius: 14px;
        box-shadow: 0 0 12px rgba(0,0,0,0.2);
        width: 80%;
        text-align: center;
        border: 2px solid;
    }

    .original {
        background-color: #2e7d32;
        border-color: #1b5e20;
        color: white;
    }

    .recaptured {
        background-color: #ef6c00;
        border-color: #bf360c;
        color: white;
    }

    .column-divider {
        border-left: 2px solid #777;
        height: auto;
        margin-top: 60px;
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

# Helper function
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# Layout with middle divider
left_col, mid_col, right_col = st.columns([0.6, 0.02, 0.38])

# ----- LEFT PANEL -----
with left_col:
    st.markdown('<div class="left-align-block">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a QR Code image", type=["jpg", "jpeg", "png"])
    image_pil = None
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        resized = image_pil.copy()
        resized.thumbnail((400, 400))
        img_base64 = get_image_base64(resized)
        st.markdown(
            f"<div style='text-align: center;'><img src='data:image/jpeg;base64,{img_base64}' style='border-radius: 10px; margin-top: 10px;'/></div>",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ----- MIDDLE DIVIDER -----
with mid_col:
    st.markdown('<div class="column-divider"></div>', unsafe_allow_html=True)

# ----- RIGHT PANEL -----
with right_col:
    st.markdown('<div class="right-align-block">', unsafe_allow_html=True)
    verify_button = st.button("🔍 Verify QR")
    
    if verify_button:
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
    st.markdown('</div>', unsafe_allow_html=True)
