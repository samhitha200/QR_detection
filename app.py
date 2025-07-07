import streamlit as st
import numpy as np
import cv2
from joblib import load
from feature_extractor import extract_white_area_features

# Load the model
model = load("rf_white_features.pkl")

# UI
st.set_page_config(page_title="QR Code Authenticity Validator", layout="centered")
st.title("QR Code Authenticity Validator")
st.caption("Distinguish between Original vs Recaptured QR codes")

# File Upload
uploaded_file = st.file_uploader("Upload a QR Code image (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        st.image(image, channels="BGR", caption="Uploaded QR Code", use_column_width=True)

        # Feature extraction
        features = extract_white_area_features(image)

        if features is not None:
            prediction = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            label = "Original" if prediction == 0 else "Recaptured"
            confidence = np.max(proba) * 100

            st.success(f"Predicted Label: {label}")
            st.info(f"Prediction Confidence: {confidence:.2f}%")
        else:
            st.warning("Not enough white area detected in the image to extract features.")
    else:
        st.error("Could not decode image. Please upload a valid file.")
