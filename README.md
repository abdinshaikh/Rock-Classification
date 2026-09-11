# Rock Classification Using Deep Learning

## Overview

This project classifies rock images into seven categories using deep learning. A CNN model was first developed as a baseline, followed by transfer learning with VGG16. The VGG16 model was fine-tuned on the rock image dataset to improve classification performance.

The project also includes an interactive prediction system that allows users to upload a rock image and receive the predicted rock type, confidence score, and Top-3 class probabilities.

## Dataset

The dataset contains 2,077 valid rock images distributed across seven classes:

- Basalt
- Coal
- Granite
- Limestone
- Marble
- Quartzite
- Sandstone

## Methodology

The project follows a two-stage modeling approach:

1. **CNN Baseline**
   - Built a convolutional neural network as a baseline for rock image classification.
   - Applied image resizing, rescaling, and data augmentation.
   - Evaluated the model on the test set.

2. **VGG16 Transfer Learning**
   - Used VGG16 pretrained on ImageNet as the feature extractor.
   - Added a custom classification head for the seven rock classes.
   - Fine-tuned the deeper convolutional layers of VGG16 using a low learning rate.
   - Evaluated the final model using accuracy, precision, recall, F1-score, and a confusion matrix.

The VGG16 fine-tuned model was selected as the final model for the prediction system.

## Data Preparation

- Validated the image dataset and removed invalid or unreadable images.
- Resized images to 256 × 256 pixels and normalized pixel values.
- Used stratified train-validation-test splitting to maintain class distribution.
- Applied random horizontal/vertical flips and rotations for data augmentation.

## Model Architecture

The final model uses VGG16 pretrained on ImageNet with the original classification layers removed. A custom classification head was added consisting of:

- Global Average Pooling
- Dense layer with 256 units and ReLU activation
- Dropout with a rate of 0.5
- Softmax output layer with 7 classes

The initial training stage kept the VGG16 convolutional base frozen. The deeper convolutional layers were then unfrozen and fine-tuned using a learning rate of 1e-5.

## Results

The final VGG16 fine-tuned model achieved:

- **Test Accuracy:** 73.07%
- **Test Loss:** 0.92
- **Macro F1-Score:** 0.67
- **Weighted F1-Score:** 0.74

The model performed particularly well on Coal, while Basalt and Granite were more challenging classes due to lower recall and precision.

## Prediction System

An interactive prediction system was built using the final fine-tuned VGG16 model. Users can upload an unseen rock image, which is processed and passed through the model to display:

- Predicted rock type
- Prediction confidence
- Top-3 class probabilities
- Confidence level based on prediction confidence and the gap between the top predictions

## Project Structure

```text
AI-Based-Rock-Classification/
├── Facies_classifier.ipynb
├── rock_predictor.py
├── README.md
├── requirements.txt
└── .gitignore

