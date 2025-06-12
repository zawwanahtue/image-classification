from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import cv2
import os
from datetime import datetime
from .image import predict_img
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('torch').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Define router
router = APIRouter(prefix="/api/v2/classify")

# Video folder
videos_folder = os.path.join(os.path.dirname(__file__), '..', r"C:\Users\soe\Downloads\video testing-20250609T043256Z-1-001\video testing")

# Unsafe labels
UNSAFE_LABELS = {
    'Adults': 'Adult',
    'Gambling': 'Gambling',
    'Political': 'Political',
    'Violence': 'Violence',
}

# Safe labels
SAFE_LABELS = {
    "Culture": "Culture",
    "Entertainment": "Entertainment",
    "Environment": "Environment",
    "Product": "Product",
    "Sports": "Sports",
    "Social": "Social",
    "Technology": "Technology",
}

# Pydantic model for JSON input
class VideoInput(BaseModel):
    video_guid: List[str]

def extract_frames_and_analyze(fileName: str) -> dict:
    """
    Video Classification Function
    """
    try:
        directory = videos_folder
        base_filename = fileName

        actual_filename = None
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": 4000,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Directory not found: {directory}",
                    "data": []
                }
            )

        for file in os.listdir(directory):
            if file.lower() == base_filename.lower() + ".mp4":
                actual_filename = file
                break
            elif file.lower().startswith(base_filename.lower()) and any(
                file.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
            ):
                actual_filename = file

        if not actual_filename:
            logger.error(f"No video file found matching: {base_filename}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": 4000,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"No video file found matching: {base_filename}",
                    "data": []
                }
            )

        full_video_path = os.path.join(directory, actual_filename)
        video = cv2.VideoCapture(full_video_path)
        if not video.isOpened():
            logger.error(f"Could not open video file: {full_video_path}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": 4000,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Could not open video file: {full_video_path}",
                    "data": []
                }
            )

        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = frame_count // (fps * 60) if fps > 0 else 0
        frame_interval = fps * 1 if fps > 0 else 1

        frame_number = 0
        processed_frames = 0
        safe_count = 0
        unsafe_count = 0
        detected_contents = []

        while frame_number < frame_count:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_number}")
                break

            success, encoded_image = cv2.imencode(".png", frame)
            if not success:
                logger.warning(f"Failed to encode frame {frame_number}")
                continue

            image_bytes = encoded_image.tobytes()
            predictions, top_class, top_confidence, frame_is_safe, frame_detected_content = predict_img(
                f"frame_{processed_frames}", image_bytes=image_bytes)

            # Ensure detected content is a string
            detected_content_value = (
                frame_detected_content[0] if isinstance(frame_detected_content, list) else frame_detected_content
            )
            detected_contents.append(detected_content_value)

            if frame_is_safe:
                safe_count += 1
            else:
                unsafe_count += 1

            processed_frames += 1
            frame_number += frame_interval

        video.release()
        cv2.destroyAllWindows()

        if processed_frames == 0:
            detected_content = ["Unknown"]
            is_safe = False
        elif safe_count == processed_frames:
            is_safe = True
            most_common = max(
                set(detected_contents), key=detected_contents.count, default="Unknown"
            )
            detected_content = [most_common]
        elif unsafe_count == processed_frames:
            is_safe = False
            most_common = max(
                set(detected_contents), key=detected_contents.count, default="Unknown"
            )
            detected_content = [most_common]
        elif safe_count > unsafe_count:
            is_safe = True
            most_common = max(
                [content for content in detected_contents if content in SAFE_LABELS.values()],
                key=detected_contents.count,
                default="Unknown"
            )
            detected_content = [most_common]
        else:
            is_safe = False
            most_common = max(
                [content for content in detected_contents if content in UNSAFE_LABELS.values()],
                key=detected_contents.count,
                default="Unknown"
            )
            detected_content = [most_common]

        return {
            "filename": fileName,
            "is_safe": is_safe,
            "content_type": detected_content,  # Return as list
            "frames_processed": processed_frames,
            "video_duration_minutes": duration_minutes
        }

    except HTTPException as e:
        logger.error(f"HTTP error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Failed to extract frame and analyze video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": 5000,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"Failed to extract frame and analyze video: {str(e)}",
                "data": []
            }
        )

@router.post("/video")
async def analyze_video(input_data: VideoInput):
    try:
        results = []

        for video in input_data.video_guid:
            result = extract_frames_and_analyze(video)
            results.append({
                "filename": video,
                "is_safe": result["is_safe"],
                "content_type": result["content_type"]  # Already a list
            })

        return {
            "error": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "Successfully classified video",
            "data": results
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": 5000,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"Video analysis error: {str(e)}",
                "data": []
            }
        )