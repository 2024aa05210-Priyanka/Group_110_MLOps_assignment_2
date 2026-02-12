import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = "D:/BITS/Semester_3/MLOps/Assignment/Assignment_2/data/raw"
PROCESSED_DIR = "D:/BITS/Semester_3/MLOps/Assignment/Assignment_2/data/processed"
IMG_SIZE = (224, 224)
SEED = 42

# Map raw folder names → normalized class names
CLASS_MAP = {
    "Cat": "cats",
    "Dog": "dogs"
}

def split_images(images, seed=42):
    """
    Split image paths into train/val/test (80/10/10)
    """
    train_imgs, temp_imgs = train_test_split(
        images, test_size=0.2, random_state=seed
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5, random_state=seed
    )
    return train_imgs, val_imgs, test_imgs


def prepare_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASS_MAP.values():
            os.makedirs(
                os.path.join(PROCESSED_DIR, split, cls),
                exist_ok=True
            )

def process_and_save(images, cls, split):
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        filename = os.path.basename(img_path)
        save_path = os.path.join(PROCESSED_DIR, split, cls, filename)
        img.save(save_path)

def preprocess():
    random.seed(SEED)
    prepare_dirs()

    for raw_cls, out_cls in CLASS_MAP.items():
        cls_dir = os.path.join(RAW_DIR, raw_cls)

        images = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        train_imgs, temp_imgs = train_test_split(
            images, test_size=0.2, random_state=SEED
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5, random_state=SEED
        )

        process_and_save(train_imgs, out_cls, "train")
        process_and_save(val_imgs, out_cls, "val")
        process_and_save(test_imgs, out_cls, "test")

    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
