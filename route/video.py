from fastapi import APIRouter, HTTPException
from pydantic import RootModel
from typing import List
import cv2
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from .image import predict_img
import logging
logging.getLogger('torch').setLevel(logging.ERROR)

# Define router
router = APIRouter()

# Video and frames folder
videos_folder = os.path.join(os.path.dirname(__file__), '..', "mnt/video-contents")

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
    'Sports': 'sports',
    'Technology': 'technology'
}

# Pydantic model for JSON input
class VideoInput(RootModel[List[str]]):
    pass

def extract_frames_and_analyze(fileName: str) -> dict:
    try:
        directory = videos_folder
        base_filename = fileName

        actual_filename = None
        if not os.path.exists(directory):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": 4001,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Directory not found: {directory}",
                    "data": []
                }
            )

        for file in os.listdir(directory):
            if file.lower().startswith(base_filename.lower()):
                if any(file.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
                    actual_filename = file
                    break

        if not actual_filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": 4001,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"No valid video file found for: {base_filename}",
                    "data": []
                }
            )

        full_video_path = os.path.join(directory, actual_filename)
        video = cv2.VideoCapture(full_video_path)

        if not video.isOpened():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": 4001,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Could not open video file: {actual_filename}",
                    "data": []
                }
            )

        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = frame_count // (fps * 60)
        frame_interval = fps * 10

        frame_number = 0
        processed_frames = 0
        safe_count = 0
        unsafe_count = 0
        detected_contents = []

        while frame_number < frame_count:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if not ret:
                break

            success, encoded_image = cv2.imencode(".png", frame)
            if not success:
                continue

            image_bytes = encoded_image.tobytes()
            predictions, top_class, top_confidence, frame_is_safe, frame_detected_content = predict_img(
                f"frame_{processed_frames}", image_bytes=image_bytes)

            if frame_is_safe:
                safe_count += 1
            else:
                unsafe_count += 1
            detected_contents.append(frame_detected_content)

            processed_frames += 1
            frame_number += frame_interval

        video.release()
        cv2.destroyAllWindows()

        if processed_frames == 0:
            detected_content = "Unknown"
            is_safe = False
        elif safe_count == processed_frames:
            is_safe = True
            detected_content = max(set(detected_contents), key=detected_contents.count, default="Unknown")
        elif unsafe_count == processed_frames:
            is_safe = False
            detected_content = max(set(detected_contents), key=detected_contents.count, default="Unknown")
        elif safe_count > unsafe_count:
            is_safe = True
            detected_content = max([content for content in detected_contents if content in SAFE_LABELS.values()],
                                   key=detected_contents.count, default="Unknown")
        else:
            is_safe = False
            detected_content = max([content for content in detected_contents if content in UNSAFE_LABELS.values()],
                                   key=detected_contents.count, default="Unknown")

        return {
            "filename": fileName,
            "is_safe": is_safe,
            "detected_content": detected_content,
            "frames_processed": processed_frames,
            "video_duration_minutes": duration_minutes
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": 5001,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"Failed to extract frame and analyze video: {e}",
                "data": []
            }
        )

@router.post("/analyze/")
async def analyze_video(input_data: VideoInput):
    try:
        results = []

        for video in input_data.root:
            result = extract_frames_and_analyze(video)
            results.append({
                "filename": video,
                "is_safe": result["is_safe"],
                "content_type": result["detected_content"]
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
                "message": f"Video analysis error: {e}",
                "data": []
            }
        )