import os
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "D:/BITS/Semester_3/MLOps/Assignment/Assignment_2/data/processed"
MODEL_DIR = "D:/BITS/Semester_3/MLOps/Assignment/Assignment_2/models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_baseline.h5")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train():
    mlflow.set_experiment("cats_dogs_baseline")

    with mlflow.start_run():
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )

        val_gen = ImageDataGenerator(rescale=1./255)

        train_data = train_gen.flow_from_directory(
            os.path.join(DATA_DIR, "train"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            seed=SEED
        )

        val_data = val_gen.flow_from_directory(
            os.path.join(DATA_DIR, "val"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            seed=SEED
        )

        model = build_model()

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS
        )

        # Log metrics
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])

        # Save and log model
        model.save(MODEL_PATH)
        mlflow.keras.log_model(model, "model")

        # Confusion matrix on validation set
        val_data.reset()
        preds = (model.predict(val_data) > 0.5).astype(int)
        y_true = val_data.classes

        cm = confusion_matrix(y_true, preds)

        plt.figure(figsize=(5, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        print(" MLflow run completed")

if __name__ == "__main__":
    train()
