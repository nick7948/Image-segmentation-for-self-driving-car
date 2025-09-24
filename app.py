"""
Flask API for Image Segmentation for Self-Driving Cars
Provides REST endpoints for image segmentation predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import os
import sys
import traceback

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.inference_service import ModelInferenceService

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global inference service
inference_service = None

def initialize_service():
    """Initialize the model inference service."""
    global inference_service
    try:
        # Try to load a model if available, otherwise use uninitialized model for demo
        model_path = os.getenv('MODEL_PATH', None)  # Can be set via environment variable
        inference_service = ModelInferenceService(model_path=model_path)
        print("Inference service initialized successfully")
    except Exception as e:
        print(f"Error initializing inference service: {e}")
        return False
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'image-segmentation-api',
        'model_loaded': inference_service is not None
    })

@app.route('/predict', methods=['POST'])
def predict_segmentation():
    """
    Predict segmentation mask for uploaded image.
    
    Expected request:
    - Content-Type: multipart/form-data
    - File field: 'image'
    
    OR
    
    - Content-Type: application/json
    - JSON body: {'image': '<base64_encoded_image>'}
    """
    try:
        if inference_service is None:
            return jsonify({
                'error': 'Inference service not initialized',
                'success': False
            }), 500
        
        image = None
        
        # Handle multipart/form-data (file upload)
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected',
                    'success': False
                }), 400
            
            try:
                image = Image.open(file.stream)
            except Exception as e:
                return jsonify({
                    'error': f'Invalid image file: {str(e)}',
                    'success': False
                }), 400
        
        # Handle JSON with base64 image
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({
                    'error': 'Missing "image" field in JSON request',
                    'success': False
                }), 400
            
            try:
                # Decode base64 image
                image_data = base64.b64decode(data['image'])
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                return jsonify({
                    'error': f'Invalid base64 image: {str(e)}',
                    'success': False
                }), 400
        
        else:
            return jsonify({
                'error': 'No image provided. Use multipart/form-data or JSON with base64 image.',
                'success': False
            }), 400
        
        # Perform prediction
        try:
            result = inference_service.predict(image)
            
            # Convert colored mask to base64 for response
            mask_base64 = inference_service.mask_to_base64(result['mask_colored'])
            
            return jsonify({
                'success': True,
                'result': {
                    'confidence': result['confidence'],
                    'mask_shape': result['shape'],
                    'mask_colored_base64': mask_base64,
                    'classes_detected': len(set([item for sublist in result['mask'] for item in sublist]))
                },
                'message': 'Segmentation completed successfully'
            })
            
        except Exception as e:
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
    
    except Exception as e:
        return jsonify({
            'error': f'Request processing failed: {str(e)}',
            'success': False,
            'traceback': traceback.format_exc()
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if inference_service is None:
        return jsonify({
            'error': 'Inference service not initialized',
            'success': False
        }), 500
    
    try:
        return jsonify({
            'success': True,
            'model_info': {
                'device': str(inference_service.device),
                'input_size': [192, 256],  # Height, Width
                'num_classes': 34,
                'model_type': 'U-Net'
            }
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to get model info: {str(e)}',
            'success': False
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Image Segmentation API for Self-Driving Cars',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /': 'API information',
            'GET /model/info': 'Model information',
            'POST /predict': 'Predict segmentation mask for image'
        },
        'usage': {
            'predict_multipart': 'POST /predict with multipart/form-data and "image" file field',
            'predict_json': 'POST /predict with JSON {"image": "base64_encoded_image"}'
        }
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({
        'error': 'File too large. Please upload a smaller image.',
        'success': False
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Internal server error occurred',
        'success': False
    }), 500

if __name__ == '__main__':
    # Initialize the inference service
    if not initialize_service():
        print("Failed to initialize inference service. Exiting.")
        sys.exit(1)
    
    # Run the Flask app
    print("Starting Image Segmentation API...")
    print("Available endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  GET  /model/info - Model information")
    print("  POST /predict   - Predict segmentation")
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )