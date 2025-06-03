import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import RootModel
from typing import List
import os
from datetime import datetime
import cv2
import logging
import torch.nn as nn
logging.getLogger('torch').setLevel(logging.ERROR)

# Define router
router = APIRouter()

# Model path and images folder
model_path = os.path.join(os.path.dirname(__file__), '..', 'efficientnetb3_70k_v2_model.pt')
images_folder = os.path.join(os.path.dirname(__file__), '..', r"C:\Users\User\Desktop\Exter")
# Create images folder if it doesn't exist

# Load model
try:
    # Initialize EfficientNetB3 with custom classifier
    model = models.efficientnet_b3(weights=None)  # Avoid deprecation warning
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),              # Standard EfficientNetB3 dropout
        nn.Linear(1536, 128),           # Matches classifier.1
        nn.SiLU(),                      # Common activation
        nn.Dropout(p=0.4),              # Additional dropout
        nn.Linear(128, 11)              # Matches classifier.4, outputs 10 classes
    )
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded as PyTorch model")
except Exception as e:
    raise ValueError(f"Cannot load model: {e}")

# Class labels dictionary
class_names = {
    0: 'Adults',
    1: 'Culture',
    2: 'Entertainment',
    3: 'Environment',
    4: 'Gambling',
    5: 'Political',
    6: 'Product',
    7: 'Social',
    8: 'Sports',
    9: 'Technology',
    10: 'Violence'
}

# Unsafe labels
UNSAFE_LABELS = {
    'Adults': 'adult',
    'Gambling': 'gambling',
    'Political': 'political',
    'Violence': 'violence',
}

# Safe labels
SAFE_LABELS = {
    'Culture': 'culture',
    'Entertainment': 'entertainment',
    'Environment': 'environment',
    'Product': 'product',
    'Social': 'social',
    'Sports': 'sports',
    'Technology': 'technology'
}

# Combine SAFE and UNSAFE labels for mapping
ALL_LABELS = {**SAFE_LABELS, **UNSAFE_LABELS}

# Pydantic model for JSON input
class ImageInput(RootModel[List[str]]):
    pass

def predict_img(fileName: str, image_bytes: bytes = None):
    try:
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            supported_extensions = ['.jpg', '.jpeg', '.png']
            image_path = None
            for ext in supported_extensions:
                potential_path = os.path.join(images_folder, f"{fileName}{ext}")
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break

            if not image_path:
                raise HTTPException(status_code=404,
                                    detail=f"Image {fileName} not found in {images_folder} with extensions {supported_extensions}")

            image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            preds = model(image)
            preds = torch.softmax(preds, dim=1)[0]

        predictions = {
            class_names[i]: float(f"{prob * 100:.2f}")
            for i, prob in enumerate(preds)
        }

        top_idx = int(torch.argmax(preds))
        top_class = class_names[top_idx]
        top_confidence = float(f"{preds[top_idx] * 100:.2f}")

        is_safe = top_class not in UNSAFE_LABELS
        detected_content = ALL_LABELS.get(top_class, "Unknown")

        return predictions, top_class, top_confidence, is_safe, detected_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@router.post("/predict/")
async def predict(input_data: ImageInput):
    try:
        results = []

        for image in input_data.root:
            predictions, top_class, top_confidence, is_safe, detected_content = predict_img(image)
            
            results.append({
                "filename": image,
                "is_safe": is_safe,
                "content_type": detected_content
            })

        return results
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": 5001,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"Prediction error: {e}",
                "data": []
            }
        )