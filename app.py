import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image
from feature_extractor import extract_white_area_features
from io import BytesIO
from fpdf import FPDF
import base64

# Load model
model = load("rf_white_features.pkl")

# Page config
st.set_page_config(page_title="QR Code Authenticity Validator", layout="wide")

# Light/Dark theme toggle
theme = st.radio("Select Theme", ["Light", "Dark"], horizontal=True)

# Custom CSS for themes
light_css = """
<style>
.block-container { padding: 1rem 2rem; }
h1 { font-size: 1.6rem; text-align: center; }
.result-card {
    padding: 1rem;
    border-radius: 12px;
    color: white;
    font-weight: bold;
    text-align: center;
    font-size: 1.3rem;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
    margin-top: 1rem;
}
.original { background-color: #2e7d32; }
.recaptured { background-color: #ef6c00; }
</style>
"""

dark_css = """
<style>
.block-container { padding: 1rem 2rem; background-color: #1e1e1e; color: white; }
h1 { font-size: 1.6rem; text-align: center; color: #f1f1f1; }
.result-card {
    padding: 1rem;
    border-radius: 12px;
    color: white;
    font-weight: bold;
    text-align: center;
    font-size: 1.3rem;
    box-shadow: 0 0 10px rgba(255,255,255,0.2);
    margin-top: 1rem;
}
.original { background-color: #388e3c; }
.recaptured { background-color: #f57c00; }
</style>
"""

st.markdown(light_css if theme == "Light" else dark_css, unsafe_allow_html=True)

# Title
st.markdown("<h1>QR Code Authenticity Validator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Distinguish between Original vs Recaptured QR codes</p>", unsafe_allow_html=True)

# Helpers
def get_image_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

def generate_pdf_report(image_pil, label, confidence, index):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="QR Code Verification Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    # Save image temporarily
    img_path = f"temp_img_{index}.jpg"
    image_pil.save(img_path)
    pdf.image(img_path, x=10, y=50, w=100)
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# File upload
uploaded_files = st.file_uploader("Upload one or more QR Code images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Process each file
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.subheader(f"Image {idx+1}")
            image_pil = Image.open(uploaded_file).convert("RGB")
            resized = image_pil.copy()
            resized.thumbnail((400, 400))
            img_base64 = get_image_base64(resized)
            st.markdown(
                f"<div style='text-align: center;'><img src='data:image/jpeg;base64,{img_base64}' style='border-radius: 10px;'/></div>",
                unsafe_allow_html=True
            )

        with col2:
            verify_button = st.button(f"🔍 Verify QR - Image {idx+1}")

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
                            <span style='font-size: 0.9rem;'>Confidence: {confidence:.2f}%</span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Download PDF report
                    pdf_file = generate_pdf_report(image_pil, label, confidence, idx)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_file,
                        file_name=f"QR_Report_{idx+1}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning("⚠️ Could not extract white area features.")
