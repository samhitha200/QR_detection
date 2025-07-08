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
    .pane-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
        padding-top: 40px;
    }

    .upload-preview {
        text-align: center;
        margin-top: 20px;
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
        padding: 1.2rem 2rem;
        font-size: 1.1rem;
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
    </style>
""", unsafe_allow_html=True)

# Helper to convert image to base64
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# ---- Layout: Two-Pane Split ----
left_col, right_col = st.columns([0.6, 0.4])

# ----- LEFT PANE -----
with left_col:
    st.markdown('<div class="pane-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload QR Code Image", type=["jpg", "jpeg", "png"])
    image_pil = None

    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        resized = image_pil.copy()
        resized.thumbnail((400, 400))
        img_base64 = get_image_base64(resized)
        st.markdown(f"""
            <div class='upload-preview'>
                <img src='data:image/jpeg;base64,{img_base64}' style='border-radius: 10px;'/>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----- RIGHT PANE -----
with right_col:
    st.markdown('<div class="pane-container">', unsafe_allow_html=True)
    verify_button = st.button("🔍 Verify QR")

    if verify_button:
        if image_pil is not None:
            image_np = np.array(image_pil)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            features = extract_white_area_features(image_cv2)

            if features is not None:
                prediction = model.predict([features])[0]
                proba = model.predict_proba([features])[0]
                label = "ORIGINAL" if prediction == 0 else "RECAPTURED"
                confidence = np.max(proba) * 100

                card_class = "original" if label == "ORIGINAL" else "recaptured"
                st.markdown(f"""
                    <div class='result-card {card_class}'>
                        {label}<br/>
                        <span style='font-size: 0.85rem;'>Confidence: {confidence:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Feature extraction failed.")
        else:
            st.warning("⚠️ Please upload an image first.")
    st.markdown('</div>', unsafe_allow_html=True)
