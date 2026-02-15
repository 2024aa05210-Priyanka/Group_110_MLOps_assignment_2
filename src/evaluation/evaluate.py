import requests
import os

API_URL = "http://localhost:8000/predict"
TEST_FOLDER = "data/processed/test"

correct = 0
total = 0

for label in ["cats", "dogs"]:
    folder = os.path.join(TEST_FOLDER, label)
    for img_name in os.listdir(folder)[:10]:  # small batch
        img_path = os.path.join(folder, img_name)

        with open(img_path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})

        pred = response.json()["label"]

        if pred == label[:-1]:  # cats → cat
            correct += 1

        total += 1

accuracy = correct / total
print(f"Post-deployment Accuracy: {accuracy:.4f}")
