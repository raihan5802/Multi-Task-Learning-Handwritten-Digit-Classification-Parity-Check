# Multi-Task Learning for Handwritten Digit Classification and Parity Check

This project implements a Convolutional Neural Network (CNN) using VGG-inspired blocks to perform two main tasks: digit classification and parity prediction on the MNIST dataset. Regularization techniques and data augmentation are applied to enhance model performance and generalization.

## Project Overview

### Objective
Develop a CNN model for:
1. **Digit Classification**: Classify digits from 0 to 9.
2. **Parity Prediction**: Predict whether each digit is even or odd.

The model leverages a multi-task learning approach with a shared feature extraction network and two output branches for the respective tasks.

### Dataset
**MNIST Dataset**:
- **Size**: 70,000 samples (60,000 for training, 10,000 for testing)
- **Classes**: 10 (digits 0–9)
- **Normalization**: Pixel values normalized to [0, 1]

### Related Works
1. [Multi-script Handwritten Digit Recognition Using Multi-task Learning](https://arxiv.org/abs/2106.08267v1)
2. [Handwritten Digit Classification on MNIST Dataset](https://github.com/pengfeinie/handwritten-digit-classification-thinking)

## Model Architecture

The model utilizes VGG blocks for feature extraction, with shared layers for both tasks:
- **Digit Classification Output Branch**: Predicts digits 0–9
- **Parity Prediction Output Branch**: Predicts even or odd

### Regularization
- **Dropout Layers**: Dropout rate of 0.4.
- **L2 Regularization**: Applied on convolutional layers.

### Optimization
- **Optimizer**: Adam with a learning rate of 0.001.
- **Callbacks**: Early stopping and learning rate reduction (ReduceLROnPlateau) for efficient training.

## Results

### Metrics
- **Digit Classification**:
  - **Accuracy**: 96.20%
  - **Precision**: 96.19%
  - **Recall**: 96.13%

- **Parity Prediction**:
  - **Accuracy**: 97.17%
  - **Precision**: 97.12%
  - **Recall**: 97.14%

### Training Statistics
- **Total Training Time**: 52.05 minutes
- **Average Epoch Time**: 104.1 seconds

## Future Work

Future improvements include:
1. **Model Architecture Tuning**: Experimenting with ResNet, DenseNet, and Inception.
2. **Hyperparameter Optimization**: Tuning learning rates, dropout rates, and regularization techniques.
3. **Task Expansion**: Extending to tasks like handwriting style and digit sequence recognition, and integrating additional datasets (e.g., EMNIST) to improve robustness.
