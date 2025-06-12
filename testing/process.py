import requests
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file_list(input_file_path: str, output_file_path: str, api_url: str, endpoint: str = "/analyze/"):
    """
    Process a list of filenames from input text file, send requests to FastAPI with {"video_guid": ["filename"]} payload,
    and append responses to output text file immediately after each request.

    Args:
        input_file_path (str): Path to input text file containing filenames
        output_file_path (str): Path to output text file to save responses
        api_url (str): Base URL of the FastAPI server (e.g., http://localhost:8000)
        endpoint (str): API endpoint to call (e.g., /analyze/ for videos, /predict/ for images)
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file_path):
            logger.error(f"Input file not found: {input_file_path}")
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        # Read filenames from input text file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            filenames = [line.strip() for line in f if line.strip()]
        
        if not filenames:
            logger.warning("Input file is empty")
            raise ValueError("Input file is empty")

        # Ensure output file is empty or created
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("")  # Clear the file if it exists

        # Process each filename and write response immediately
        for filename in filenames:
            logger.info(f"Processing filename: {filename}")
            try:
                # Prepare JSON payload with video_guid
                payload = {"video_guid": [filename]}
                response = requests.post(f"{api_url}{endpoint}", json=payload)
                
                # Prepare result for this filename
                result = {
                    "filename": filename,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Check response status
                if response.status_code == 200:
                    response_data = response.json()
                    # Extract data field if it exists
                    if "data" in response_data and response_data.get("error") == 0:
                        result["response"] = response_data["data"][0] if response_data["data"] else {"error": "No data in response"}
                        result["api_timestamp"] = response_data.get("timestamp", "")
                        result["message"] = response_data.get("message", "")
                    else:
                        result["response"] = {"error": response_data.get("message", "Invalid response format")}
                        result["api_timestamp"] = response_data.get("timestamp", "")
                        result["message"] = response_data.get("message", "")
                    logger.info(f"Successfully processed: {filename}")
                else:
                    logger.error(f"Failed to process {filename}: {response.status_code} - {response.text}")
                    result["response"] = {"error": response.text}
                    result["api_timestamp"] = ""
                    result["message"] = ""

                # Append result to output file immediately
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"Filename: {result['filename']}\n")
                    f.write(f"Timestamp: {result['timestamp']}\n")
                    f.write(f"API Timestamp: {result['api_timestamp']}\n")
                    f.write(f"Message: {result['message']}\n")
                    f.write(f"Response: {json.dumps(result['response'], indent=2)}\n")
                    f.write("-" * 50 + "\n")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                result = {
                    "filename": filename,
                    "response": {"error": str(e)},
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "api_timestamp": "",
                    "message": ""
                }
                # Append error result to output file immediately
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"Filename: {result['filename']}\n")
                    f.write(f"Timestamp: {result['timestamp']}\n")
                    f.write(f"API Timestamp: {result['api_timestamp']}\n")
                    f.write(f"Message: {result['message']}\n")
                    f.write(f"Response: {json.dumps(result['response'], indent=2)}\n")
                    f.write("-" * 50 + "\n")

        logger.info(f"All filenames processed. Results appended to {output_file_path}")

    except Exception as e:
        logger.error(f"Failed to process file list: {str(e)}")
        raise

    process_file_list(input_file, output_file, api_url, endpoint)
if __name__ == "__main__":
    # Example usage
    input_file = "input_filenames.txt"
    output_file = "output_results.txt"
    api_url = "http://localhost:8000"  # Replace with your FastAPI server URL
    endpoint = "/video/api/v2/classify/video/"  # Use "/predict/" for images or "/analyze/" for videos

    process_file_list(input_file, output_file, api_url, endpoint)