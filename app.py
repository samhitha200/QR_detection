import streamlit as st
import numpy as np
import cv2
from joblib import load
from feature_extractor import extract_white_area_features

# Load the trained model
model = load("rf_white_features.pkl")

# Streamlit UI settings
st.set_page_config(page_title="QR Code Authenticity Validator", layout="centered")
st.title("QR Code Authenticity Validator")
st.caption("Distinguish between Original vs Recaptured QR codes")

# File uploader for QR image
uploaded_file = st.file_uploader("Upload a QR Code image (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(image, channels="BGR", caption="Uploaded QR Code", use_column_width=True)

        # Extract features
        features = extract_white_area_features(image)

        if np.sum(features) == 0:  # Option 2: Reject if no valid white features
            st.warning("The uploaded image does not contain enough white area for a valid prediction.")
        else:
            # Predict using loaded model
            prediction = model.predict([features])[0]
            proba = model.predict_proba([features])[0]

            label = "Original" if prediction == 0 else "Recaptured"
            confidence = np.max(proba) * 100

            st.success(f"Predicted Label: {label}")
            st.info(f"Prediction Confidence: {confidence:.2f}%")
    else:
        st.error("Failed to read the image. Please upload a valid file.")
