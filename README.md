# Image-segmentation-for-self-driving-car

# Project Overview
This project uses a U-Net-based Convolutional Neural Network (CNN) to perform semantic segmentation on urban street scenes, tailored for self-driving car applications

# Objective
To build a segmentation model that effectively identifies key navigational elements in urban street images, such as roads and vehicles, to support autonomous driving perception.

# Notebook: Image_Segmentation_Self_Driving_Class_Final_Project_(2).ipynb
This notebook includes all necessary code for data loading, model building, training, evaluation, and visualization.

# Notebook Sections
Data Loading and Preprocessing: Loads and preprocesses images and masks.
Model Definition: Defines the U-Net architecture, including encoder and decoder blocks for pixel-level classification.
Training the Model: Runs model training, tracking loss over epochs.
Inference and Visualization: Displays sample input images, true masks, and predicted masks for visual analysis.

# REST API for Postman Testing

This project now includes a REST API that can be tested with Postman or any HTTP client.

## Starting the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/model/info` - Model information  
- **POST** `/predict` - Image segmentation prediction

## Testing with Postman

See [POSTMAN_API_DOCUMENTATION.md](./POSTMAN_API_DOCUMENTATION.md) for detailed instructions on:
- Setting up Postman requests
- Uploading images for segmentation
- Understanding API responses
- Error handling

### Quick Test

1. Start the API: `python app.py`
2. Open Postman
3. Create a POST request to `http://localhost:5000/predict`
4. In the Body tab, select "form-data"
5. Add a key "image" with type "File" and upload an image
6. Send the request to get segmentation results

## API Features

- **File Upload**: Upload images via multipart/form-data
- **Base64 Support**: Send images as base64-encoded JSON
- **Visual Results**: Returns color-coded segmentation masks as base64 images
- **Model Info**: Get details about the loaded U-Net model
- **Error Handling**: Comprehensive error responses
- **CORS Enabled**: Works with web applications

# Future Improvements
Enhanced Model Architectures: Explore advanced architectures like DeepLabV3+ or integrate attention mechanisms to capture finer details in complex scenes.
Real-Time Deployment: Optimize the model for faster inference, enabling real-time application in self-driving systems.
API Enhancements: Add batch processing, model version management, and authentication.
