import os
import logging
import io
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def get_model():
    """Singleton pattern for model loading"""
    global model
    if model is None:
        logger.info("Loading ResNet50 model...")
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        logger.info("Model loaded successfully")
    return model

def process_image_file(file_or_url):
    """Process an image file or URL directly."""
    try:
        logger.info(f"Processing input: {file_or_url}")
        
        # Check if the input is a URL
        if isinstance(file_or_url, str) and file_or_url.startswith("http"):
            # Fetch the image from the URL with timeout
            response = requests.get(file_or_url, timeout=30)
            response.raise_for_status()
            file_bytes = response.content
            logger.info(f"Fetched image from URL, size: {len(file_bytes)} bytes")
        else:
            # Treat it as a file object
            file_bytes = file_or_url.read()
            logger.info(f"Received file upload, size: {len(file_bytes)} bytes")
        
        # Create a BytesIO object
        img_io = io.BytesIO(file_bytes)
        
        # Open with PIL
        img = Image.open(img_io)
        logger.info(f"Original image size: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info("Converted image to RGB mode")
        
        # Resize
        img = img.resize((224, 224))
        logger.info("Resized image to 224x224")
        
        # Convert to array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Get features
        logger.info("Extracting features with ResNet50")
        features = get_model().predict(img_array)
        logger.info(f"Features shape: {features.shape}")
        
        return features
    
    except requests.Timeout:
        logger.error("Timeout while fetching image")
        return None
    except requests.RequestException as e:
        logger.error(f"Error fetching image: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return None

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "match": "/match (POST) - Compare two images"
        }
    })

@app.route('/health')
def health():
    """Detailed health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
    })

@app.route('/match', methods=['POST'])
def match():
    """Main endpoint for image comparison"""
    try:
        logger.info("Received match request")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        if 'image1' not in data or 'image2' not in data:
            return jsonify({"error": "Please provide both 'image1' and 'image2' URLs"}), 400
        
        # Log received image URLs
        logger.info(f"Processing images: {data['image1']}, {data['image2']}")
        
        # Process images
        features1 = process_image_file(data['image1'])
        features2 = process_image_file(data['image2'])
        
        if features1 is None or features2 is None:
            return jsonify({"error": "Failed to process one or both images"}), 400
        
        # Calculate similarity
        similarity = cosine_similarity(features1, features2)
        match_score = round(float(similarity[0][0]) * 100, 2)
        
        logger.info(f"Calculated match score: {match_score}")
        return jsonify({
            "match_score": match_score,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error in match endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Load model at startup
    get_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))