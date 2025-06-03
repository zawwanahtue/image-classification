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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define router
router = APIRouter()

# Model path and images folder
model_path = os.path.join(os.path.dirname(__file__), '..', 'efficientnetb3_70k_v2_model.pt')
images_folder = os.path.join(os.path.dirname(__file__), '..', "mnt/image-contents")
# Create images folder if it doesn't exist

# Load model
try:
    # Define the model architecture
    model = models.efficientnet_b3(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(1536, 128),
        torch.nn.SiLU(),
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(128, 11),
    )

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class_names = {
    0: "Adults",
    1: "Culture",
    2: "Entertainment",
    3: "Environment",
    4: "Gambling",
    5: "Political",
    6: "Product",
    7: "Social",
    8: "Sports",
    9: "Technology",
    10: "Violence",
}

UNSAFE_LABELS = {
    "Adults": "Adult",
    "Political": "Political",
    "Violence": "Violence",
    "Gambling": "Gambling",
}

SAFE_LABELS = {
    "Culture": "Culture",
    "Entertainment": "Entertainment",
    "Environment": "Environment",
    "Product": "Product",
    "Sports": "Sports",
    "Social": "Social",
    "Technology": "Technology",
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