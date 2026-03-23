import argparse
import os
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"


def predict_image(image_path: str) -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        save=True,
        show=False
    )

    print("Detection completed.")

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print("\nDetected Objects:")
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"{class_name}: {confidence * 100:.2f}%")
        else:
            print("No objects detected.")


def main():
    parser = argparse.ArgumentParser(description="Run PPE detection on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    predict_image(args.image)


if __name__ == "__main__":
    main()
