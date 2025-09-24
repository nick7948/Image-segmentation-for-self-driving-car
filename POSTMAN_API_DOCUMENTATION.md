# API Documentation for Postman

## Image Segmentation API for Self-Driving Cars

This REST API provides image segmentation capabilities using a U-Net model trained for self-driving car applications.

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
**GET** `/health`

Returns the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "service": "image-segmentation-api", 
  "model_loaded": true
}
```

#### 2. API Information
**GET** `/`

Returns information about available endpoints and usage.

**Response:**
```json
{
  "service": "Image Segmentation API for Self-Driving Cars",
  "version": "1.0.0",
  "endpoints": {
    "GET /": "API information",
    "GET /health": "Health check",
    "GET /model/info": "Model information", 
    "POST /predict": "Predict segmentation mask for image"
  },
  "usage": {
    "predict_multipart": "POST /predict with multipart/form-data and \"image\" file field",
    "predict_json": "POST /predict with JSON {\"image\": \"base64_encoded_image\"}"
  }
}
```

#### 3. Model Information
**GET** `/model/info`

Returns information about the loaded model.

**Response:**
```json
{
  "success": true,
  "model_info": {
    "device": "cpu",
    "input_size": [192, 256],
    "num_classes": 34,
    "model_type": "U-Net"
  }
}
```

#### 4. Image Segmentation Prediction
**POST** `/predict`

Performs image segmentation on the uploaded image.

##### Method 1: Multipart Form Data (File Upload)

**Content-Type:** `multipart/form-data`

**Form Fields:**
- `image`: Image file (JPG, PNG, etc.)

**Postman Setup:**
1. Set method to POST
2. Set URL to `http://localhost:5000/predict`
3. In the Body tab, select "form-data"
4. Add key "image" with type "File"
5. Upload your image file

##### Method 2: JSON with Base64 Image

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "image": "<base64_encoded_image_data>"
}
```

**Postman Setup:**
1. Set method to POST
2. Set URL to `http://localhost:5000/predict`
3. In the Body tab, select "raw" and "JSON"
4. Add the JSON payload with base64-encoded image

**Success Response:**
```json
{
  "success": true,
  "result": {
    "confidence": 0.85,
    "mask_shape": [192, 256],
    "mask_colored_base64": "<base64_encoded_colored_mask>",
    "classes_detected": 12
  },
  "message": "Segmentation completed successfully"
}
```

**Error Response:**
```json
{
  "error": "Error message",
  "success": false
}
```

### Testing with Postman

#### Prerequisites
1. Install Postman
2. Start the API server: `python app.py`
3. Ensure the server is running on `http://localhost:5000`

#### Test Collection Steps

1. **Import or Create Collection**
   - Create a new collection named "Image Segmentation API"

2. **Add Environment Variables** (optional)
   - Create environment with base URL: `{{base_url}}` = `http://localhost:5000`

3. **Test Requests in Order:**

   a. **Health Check**
      - GET `{{base_url}}/health`
      - Should return status "healthy"

   b. **API Info**
      - GET `{{base_url}}/`
      - Should return service information

   c. **Model Info**
      - GET `{{base_url}}/model/info`
      - Should return model details

   d. **Image Prediction (File Upload)**
      - POST `{{base_url}}/predict`
      - Body: form-data with "image" file field
      - Upload a test image (JPG/PNG)
      - Should return prediction results

   e. **Image Prediction (Base64)**
      - POST `{{base_url}}/predict`
      - Body: raw JSON with base64 image
      - Should return prediction results

### Sample Test Images

For testing purposes, you can use:
- Street scene images
- Traffic images  
- Urban environment photos
- Any RGB image (will be resized to 192x256)

### Error Handling

Common error responses:
- `400`: Bad request (no image provided, invalid image format)
- `413`: File too large
- `500`: Server error (model loading issues, prediction failure)

### Notes

- Images are automatically resized to 192x256 pixels for model input
- The model outputs segmentation masks with 34 different classes
- Confidence scores represent the average probability across all pixels
- Colored masks are provided as base64-encoded PNG images for visualization
- This is a development server - use a production WSGI server for deployment