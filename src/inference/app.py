import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf

IMG_SIZE = (224, 224)

# Use TensorFlow SavedModel (directory, not .h5)
MODEL_PATH = os.path.join("models", "cnn_baseline_tf")

app = FastAPI(title="Cats vs Dogs Classifier")

model = None  # IMPORTANT: do not load at import time


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.on_event("startup")
def load_model():
    global model
    model = tf.saved_model.load(MODEL_PATH)
    print("✅ Model loaded successfully")



@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    input_tensor = preprocess_image(image)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    # Call SavedModel signature
    infer = model.signatures["serving_default"]
    outputs = infer(input_tensor)

    # Get first output tensor
    prediction = float(list(outputs.values())[0][0][0])

    label = "dog" if prediction > 0.5 else "cat"
    confidence = prediction if label == "dog" else 1 - prediction

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }
