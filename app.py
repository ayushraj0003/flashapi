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
import gc

# Configure memory growth and disable GPU if not needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Optimize TensorFlow memory usage
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable with lazy loading
model = None

def get_model():
    """Singleton pattern for model loading with memory optimization"""
    global model
    if model is None:
        logger.info("Loading ResNet50 model...")
        try:
            # Use mixed precision for better memory efficiency
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            model = ResNet50(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
            # Force model to build
            model.predict(np.zeros((1, 224, 224, 3)))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    return model

def cleanup_memory():
    """Helper function to clean up memory"""
    gc.collect()
    tf.keras.backend.clear_session()

def process_image_file(file_or_url):
    """Process an image file or URL with memory optimization"""
    try:
        logger.info(f"Processing input: {file_or_url}")
        
        # Check if the input is a URL
        if isinstance(file_or_url, str) and file_or_url.startswith("http"):
            response = requests.get(file_or_url, timeout=10, stream=True)
            response.raise_for_status()
            file_bytes = response.content
            logger.info(f"Fetched image from URL, size: {len(file_bytes)} bytes")
        else:
            file_bytes = file_or_url.read()
            logger.info(f"Received file upload, size: {len(file_bytes)} bytes")
        
        # Process image in memory-efficient way
        with io.BytesIO(file_bytes) as img_io:
            with Image.open(img_io) as img:
                logger.info(f"Original image size: {img.size}, mode: {img.mode}")
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize with LANCZOS for better quality/memory trade-off
                img = img.resize((224, 224), Image.LANCZOS)
                
                # Convert to array efficiently
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Get features
                logger.info("Extracting features with ResNet50")
                features = get_model().predict(img_array, batch_size=1)
                
                # Clean up
                del img_array
                cleanup_memory()
                
                return features
    
    except requests.Timeout:
        logger.error("Timeout while fetching image")
        return None
    except requests.RequestException as e:
        logger.error(f"Error fetching image: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        cleanup_memory()
        return None

@app.route('/match', methods=['POST'])
def match():
    """Main endpoint for image comparison with memory management"""
    try:
        logger.info("Received match request")
        
        data = request.get_json()
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({"error": "Please provide both 'image1' and 'image2' URLs"}), 400
        
        logger.info(f"Processing images: {data['image1']}, {data['image2']}")
        
        # Process images sequentially to manage memory
        features1 = process_image_file(data['image1'])
        if features1 is None:
            return jsonify({"error": "Failed to process first image"}), 400
            
        features2 = process_image_file(data['image2'])
        if features2 is None:
            return jsonify({"error": "Failed to process second image"}), 400
        
        # Calculate similarity
        similarity = cosine_similarity(features1, features2)
        match_score = round(float(similarity[0][0]) * 100, 2)
        
        # Clean up
        del features1, features2
        cleanup_memory()
        
        logger.info(f"Calculated match score: {match_score}")
        return jsonify({
            "match_score": match_score,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error in match endpoint: {str(e)}", exc_info=True)
        cleanup_memory()
        return jsonify({"error": str(e)}), 500

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
    memory_info = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = {
            "memory_percent": process.memory_percent(),
            "memory_info": str(process.memory_info())
        }
    except ImportError:
        memory_info = {"error": "psutil not installed"}

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
        "memory_info": memory_info
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=False)