from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import logging
import requests  # Import requests to fetch images from URLs

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

import requests

def process_image_file(file_or_url):
    """Process an image file or URL directly."""
    try:
        logger.debug(f"Processing input: {file_or_url}")
        
        # Check if the input is a URL
        if isinstance(file_or_url, str) and file_or_url.startswith("http"):
            # Fetch the image from the URL
            response = requests.get(file_or_url)
            response.raise_for_status()  # Raise an error for bad responses
            file_bytes = response.content
            logger.debug(f"Fetched image from URL, size: {len(file_bytes)} bytes")
        else:
            # Treat it as a file object
            file_bytes = file_or_url.read()
            logger.debug(f"Received file upload, size: {len(file_bytes)} bytes")
        
        # Create a BytesIO object
        img_io = io.BytesIO(file_bytes)
        
        # Open with PIL
        img = Image.open(img_io)
        logger.debug(f"Original image size: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.debug("Converted image to RGB mode")
        
        # Resize
        img = img.resize((224, 224))
        logger.debug("Resized image to 224x224")
        
        # Convert to array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Get features
        logger.debug("Extracting features with ResNet50")
        features = model.predict(img_array)
        logger.debug(f"Features shape: {features.shape}")
        
        return features
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return None

def fetch_image_from_url(url):
    """Fetch an image from a URL and return its processed features."""
    try:
        logger.debug(f"Fetching image from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        logger.debug(f"Fetched image from URL: {url}")

        # Process the fetched image
        return process_image_file(io.BytesIO(response.content))
    except Exception as e:
        logger.error(f"Error fetching image from URL: {str(e)}", exc_info=True)
        return None

@app.route('/match', methods=['POST'])
def match():
    try:
        logger.debug("Received match request")
        
        data = request.get_json()
        if 'image1' not in data or 'image2' not in data:
            logger.error("Missing image URLs in request")
            return jsonify({"error": "Please provide both images"}), 400
        
        # Log received image URLs
        logger.debug(f"Image1 URL: {data['image1']}")
        logger.debug(f"Image2 URL: {data['image2']}")
        
        # Process images
        features1 = process_image_file(data['image1'])
        features2 = process_image_file(data['image2'])
        
        if features1 is None or features2 is None:
            logger.error("Failed to process one or both images")
            return jsonify({"error": "Failed to process images"}), 400
        
        # Calculate similarity
        similarity = cosine_similarity(features1, features2)
        match_score = round(float(similarity[0][0]) * 100, 2)
        
        logger.debug(f"Calculated match score: {match_score}")
        return jsonify({"match_score": match_score})
    
    except Exception as e:
        logger.error(f"Error in match endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
