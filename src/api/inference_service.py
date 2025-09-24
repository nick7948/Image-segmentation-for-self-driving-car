"""
Inference service for image segmentation model.
Provides functionality to load a trained model and perform predictions.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os

from src.models.unet import UNet


class ModelInferenceService:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the inference service.
        
        Args:
            model_path: Path to the trained model file (optional)
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
        ])
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize a model (for demonstration purposes)
            self.model = UNet(in_ch=3, n_filters=64, n_classes=34).to(self.device)
            print("Warning: Using uninitialized model. Load a trained model for actual predictions.")

    def load_model(self, model_path):
        """Load a trained model from file."""
        try:
            self.model = UNet(in_ch=3, n_filters=64, n_classes=34).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Initialize new model as fallback
            self.model = UNet(in_ch=3, n_filters=64, n_classes=34).to(self.device)

    def preprocess_image(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)

    def predict(self, image):
        """
        Perform segmentation prediction on an image.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: Prediction results containing mask and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(input_tensor)
                # Apply softmax to get probabilities
                predictions = F.softmax(predictions, dim=1)
                
                # Get predicted class for each pixel
                predicted_mask = torch.argmax(predictions, dim=1)
                
                # Get confidence (max probability)
                confidence = torch.max(predictions, dim=1)[0]
                
            # Convert to numpy for easier handling
            mask = predicted_mask.cpu().numpy()[0]
            conf = confidence.cpu().numpy()[0]
            
            # Convert mask to color-coded image for visualization
            mask_colored = self.colorize_mask(mask)
            
            return {
                'mask': mask.tolist(),  # Convert numpy to list for JSON serialization
                'mask_colored': mask_colored,
                'confidence': float(conf.mean()),  # Average confidence
                'shape': mask.shape
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    def colorize_mask(self, mask):
        """
        Convert segmentation mask to color-coded image.
        
        Args:
            mask: numpy array of class predictions
            
        Returns:
            PIL Image: Color-coded segmentation mask
        """
        # Create a simple colormap (you can customize this)
        colormap = np.random.randint(0, 255, (34, 3), dtype=np.uint8)  # 34 classes
        colormap[0] = [0, 0, 0]  # Background as black
        
        # Apply colormap
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(34):
            colored_mask[mask == class_id] = colormap[class_id]
        
        return Image.fromarray(colored_mask)

    def mask_to_base64(self, mask_image):
        """Convert PIL image to base64 string."""
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode()