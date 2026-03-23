import os
from typing import List

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from utils.compliance import build_summary

MODEL_PATH = "models/best.pt"

st.set_page_config(page_title="WorkSafe Vision AI", layout="wide")

st.title("WorkSafe Vision AI")
st.subheader("Construction PPE Detection System")

st.markdown(
    """
This application detects workers and PPE items from construction site images.
It identifies Hard Hats, Safety Vests, Safety Glasses, Gloves, and Boots,
and shows a compliance summary based on visible detections.
"""
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please place best.pt inside the models folder.")
    st.stop()

model = YOLO(MODEL_PATH)

confidence = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
uploaded_file = st.file_uploader("Upload a construction image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model.predict(image, conf=confidence)
    annotated = results[0].plot()

    detected_labels: List[str] = []
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_labels.append(class_name)

    summary = build_summary(detected_labels)

    with col2:
        st.image(annotated, caption="Detection Result", use_container_width=True)

    st.markdown("## Detection Summary")
    if detected_labels:
        for label in detected_labels:
            st.write(f"- {label}")
    else:
        st.write("No objects detected.")

    st.markdown("## Compliance Status")
    st.write(f"**Status:** {summary['status']}")

    if summary["missing"]:
        st.write("**Missing PPE:**")
        for item in summary["missing"]:
            st.write(f"- {item}")
    else:
        st.write("All required PPE items were detected.")
