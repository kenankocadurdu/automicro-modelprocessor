# AutoMicro Model Processor

AutoMicro Model Processor is a containerized FastAPI-based service for training and inference of deep learning models on pathology/microbiology images. It integrates with MinIO/S3 storage and supports model selection, data augmentation, training lifecycle management, and explainability through Grad-CAM.

## Features

- **Model Training API**: Train deep learning models (e.g., ResNet, MobileNet) via REST endpoints.
- **MinIO Integration**: Load/save datasets and models directly from S3-compatible storage.
- **Patch Classification**: Predict class labels from whole slide image patches.
- **Model Explainability**: Visualize predictions using Grad-CAM.
- **Custom Hyperparameters**: Flexible training configuration with runtime parameters.
- **Training History & Metrics**: Track and export metrics, loss plots, and evaluation reports.

## Tech Stack

- **Framework**: FastAPI, Uvicorn
- **Deep Learning**: PyTorch, TorchVision
- **Storage**: MinIO (S3-compatible)
- **Visualization**: Grad-CAM, Matplotlib
- **Packaging**: Docker


