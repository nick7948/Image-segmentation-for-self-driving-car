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

# Future Improvements
Enhanced Model Architectures: Explore advanced architectures like DeepLabV3+ or integrate attention mechanisms to capture finer details in complex scenes.
Real-Time Deployment: Optimize the model for faster inference, enabling real-time application in self-driving systems.
