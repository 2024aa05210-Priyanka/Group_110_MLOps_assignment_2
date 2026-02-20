# Cats vs Dogs – End-to-End MLOps Pipeline

A production-ready MLOps pipeline for a TensorFlow-based image classification model (Cats vs Dogs), built with modern DevOps and ML engineering practices.

This project demonstrates how to move from model training to containerized deployment with CI/CD automation and monitoring.

---

## Project Overview

This repository includes:

- Image preprocessing pipeline
- CNN model training (TensorFlow)
- Experiment tracking with MLflow
- FastAPI-based inference service
- Docker containerization
- GitHub Actions CI/CD
- Deployment automation
- Basic monitoring & metrics

The system follows real-world ML system design principles.

---

##  System Architecture
```
Raw Data
↓
Preprocessing
↓
Model Training (TensorFlow)
↓
MLflow Tracking
↓
Export Model (SavedModel)
↓
FastAPI Inference Service
↓
Docker Container
↓
GitHub Actions CI/CD
↓
Deployment + Smoke Tests
```


---

##  Repository Structure
```
├── src/
│ ├── data/
│ │ └── preprocess.py
│ ├── models/
│ │ └── train.py
│ └── inference/
│ └── app.py
│
├── tests/
│ ├── test_api.py
│ └── test_preprocess.py
│
├── .github/workflows/
│ └── ci.yml
│
├── dockerfile
├── requirements.txt
├── pytest.ini
└── README.md

```
---

##  Model Details

- Framework: TensorFlow 2.13
- Architecture: Convolutional Neural Network (CNN)
- Input Shape: 224x224 RGB image
- Output: Binary classification (Cat / Dog)
- Experiment Tracking: MLflow
- Model Format: TensorFlow SavedModel

---

##  Running the Inference API

### Option 1 — Run Locally (Without Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Start server:
```
uvicorn src.inference.app:app --reload
```

## Open API docs:

http://localhost:8000/docs

## Option 2 — Run with Docker (Recommended)

Build image:
```
docker build -t cats-dogs-api .
```

Run container:
```
docker run -d -p 8000:8000 cats-dogs-api
```

Access:
```
http://localhost:8000/health
```
## API Endpoints
Health Check
GET /health


Response:

{
  "status": "ok"
}

Predict
POST /predict


## Request (multipart form):

curl -X POST "http://localhost:8000/predict" \
-F "file=@sample.jpg"


Response:

{
  "label": "dog",
  "confidence": 0.94
}

## Metrics (Monitoring)
GET /metrics


Tracks:

Total request count

Average inference latency

Example:

{
  "total_requests": 12,
  "average_latency_seconds": 0.031
}

## CI/CD Pipeline

GitHub Actions automatically performs:

Install dependencies

Run unit tests (pytest)

Build Docker image

Push image to GitHub Container Registry

Deploy container

Run smoke tests (health + prediction)

## Pipeline config:

.github/workflows/ci.yml


CI fails if:

Tests fail

Docker build fails

Deployment smoke test fails

## Testing

Run tests locally:

pytest


Tests cover:

Data preprocessing logic

API health endpoint

## Monitoring & Logging

The inference service logs:

HTTP method

Endpoint path

Response status

Request latency

Basic metrics are exposed via /metrics.

This provides lightweight observability without external monitoring systems.

## Versioning & Reproducibility

All dependencies are pinned in requirements.txt

Docker image ensures environment reproducibility

CI validates every push to master

 ## Container Registry

Docker images are published to:

ghcr.io/<username>/group_110_mlops_assignment_2/cats-dogs-api

## Key Features

Reproducible ML training workflow

Containerized inference service

Automated testing

Automated deployment

Basic monitoring implementation

Production-style project structure





