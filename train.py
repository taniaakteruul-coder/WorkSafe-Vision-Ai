import argparse
import os
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
DATA_YAML = "data.yaml"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cpu"
PROJECT_FOLDER = "runs/train"
RUN_NAME = "worksafe_vision_ai"

BEST_MODEL_PATH = os.path.join(PROJECT_FOLDER, RUN_NAME, "weights", "best.pt")


def check_dataset():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Dataset configuration file '{DATA_YAML}' was not found."
        )


def train_model():
    check_dataset()
    model = YOLO(MODEL_NAME)

    print("Starting training...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_FOLDER,
        name=RUN_NAME,
        pretrained=True,
        save=True,
        verbose=True,
        plots=True
    )

    print("Training completed.")
    print(f"Best model path: {BEST_MODEL_PATH}")


def validate_model():
    if not os.path.exists(BEST_MODEL_PATH):
        print("Best model not found. Train the model first.")
        return

    model = YOLO(BEST_MODEL_PATH)
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train WorkSafe Vision AI model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--validate", action="store_true", help="Validate the trained model")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.validate:
        validate_model()
    else:
        print("Usage:")
        print("  python train.py --train")
        print("  python train.py --validate")


if __name__ == "__main__":
    main()
