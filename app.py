from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from joblib import load

from feature_extractor import extract_white_area_features

app = FastAPI()

model = load("/content/rf_white_features.pkl")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    features = extract_white_area_features(image_cv2).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()

    return {"result": str(prediction), "confidence": float(probability)}
