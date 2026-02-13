FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.95.2 \
    uvicorn==0.22.0 \
    python-multipart==0.0.9 \
    pillow==10.2.0 \
    scikit-learn==1.3.2

COPY src ./src
#COPY models ./models
EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
