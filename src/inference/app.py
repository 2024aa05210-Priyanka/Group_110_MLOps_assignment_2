import io
import os
import time
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
import tensorflow as tf

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join("models", "cnn_baseline_tf")

app = FastAPI(title="Cats vs Dogs Classifier")

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ----------------------------
# Monitoring Metrics
# ----------------------------
request_count = 0
total_latency = 0.0

model = None


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
    logger.info("✅ Model loaded successfully")


# ----------------------------
# Middleware for monitoring
# ----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count, total_latency

    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    request_count += 1
    total_latency += latency

    logger.info(
        f"Method={request.method} "
        f"Path={request.url.path} "
        f"Status={response.status_code} "
        f"Latency={latency:.4f}s"
    )

    return response


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    return {
        "total_requests": request_count,
        "average_latency": round(avg_latency, 4)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    input_tensor = preprocess_image(image)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    infer = model.signatures["serving_default"]
    outputs = infer(input_tensor)

    prediction = float(list(outputs.values())[0][0][0])

    label = "dog" if prediction > 0.5 else "cat"
    confidence = prediction if label == "dog" else 1 - prediction

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }
