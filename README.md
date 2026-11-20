# Smart Fishing: Marine Object Classification and Telemetry System

An automated marine species classification and monitoring system developed for smart fisheries and marine conservation. The project utilizes transfer learning with a VGG-16 backbone to identify and classify underwater entities in real time from video feeds.

## System Overview

Manual visual monitoring in aquaculture and marine observation is labor-intensive and error-prone. This system provides an automated computer vision pipeline integrated with OpenCV stream processing to detect and categorize marine objects.

### Target Classes
- Fish
- Shark
- Diver
- Person
- Rocks
- Marine Trash / Debris
- Non-target Entities

## Architecture & Pipeline

1. **Feature Extraction**: Pre-trained VGG-16 backbone with ImageNet weights, freezing initial convolutional blocks to preserve low-level visual features.
2. **Classification Head**: Global flattening followed by a Dense softmax layer mapped to target marine categories.
3. **Data Augmentation**: Pipeline augmented via `ImageDataGenerator` with shear transformations, zoom variations, and horizontal flips.
4. **Real-time Stream Inference**: Ingests video stream (`cv2.VideoCapture`), resizes frames to 224x224, applies zero-centered normalization, and computes softmax class probabilities.

## Repository Structure

```
Smart-Fishing/
├── Smart Fishing Model Code.py        # Model architecture, data augmentation, and training loop
├── Smart Fishing Run Model Code.py    # Real-time inference and video capture loop
├── requirements.txt                   # Environment dependencies
└── README.md                          # Technical documentation
```

## Getting Started

### 1. Requirements
- Python 3.9+
- Webcam, USB capture card, or underwater video file

### 2. Setup
```bash
git clone https://github.com/AmjedKhaled165/Smart-Fishing.git
cd Smart-Fishing
pip install -r requirements.txt
```

### 3. Pre-trained Weights
Download the trained model file (`Smart Fishing Model.h5`) and place it in the project root:
- [Pre-trained Model Weights (Google Drive)](https://drive.google.com/drive/folders/1PA2Vb6GRDbRBK1e4pRNYyj5bvY3iIUHq?usp=sharing)

### 4. Running Inference
```bash
python "Smart Fishing Run Model Code.py"
```
Press `Esc` to stop the video stream.

### 5. Training
To retrain or fine-tune on custom marine datasets:
1. Place dataset images under `Train/` and `Test/` with labeled subdirectories.
2. Configure image directories in `Smart Fishing Model Code.py`.
3. Execute:
```bash
python "Smart Fishing Model Code.py"
```

## Technologies
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- VGG-16 Transfer Learning
