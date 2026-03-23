# WorkSafe Vision AI: Construction PPE Detection System

## Project Overview
WorkSafe Vision AI is a computer vision project developed to support safety monitoring on construction sites. The system analyses construction images to detect workers and important PPE items such as Hard Hats, Safety Vests, Safety Glasses, Gloves, and Boots.

## Features
- Detects multiple workers in an image
- Detects key PPE items
- Shows bounding boxes and confidence scores
- Generates a simple compliance summary
- Built using Python, YOLOv8, and Streamlit

## Dataset
Construction Site Safety Dataset (Kaggle)  
https://www.kaggle.com/datasets/constantinwerner/construction-site-safety-dataset

## Project Structure
- `train.py` - model training and validation
- `detect.py` - image-based prediction
- `app.py` - Streamlit live application
- `utils/compliance.py` - PPE compliance summary logic
- `data.yaml` - dataset configuration
- `models/best.pt` - trained model

## Installation
```bash
pip install -r requirements.txt
