# MNIST Classification with a Neural Network Built from Scratch

## Overview
This project implements a **fully connected neural network (MLP) from scratch using NumPy** to classify handwritten digits from the MNIST dataset.  
The goal of the project is to demonstrate a deep understanding of **neural network fundamentals**, including forward propagation, backpropagation, and optimization—without relying on high-level deep learning frameworks such as PyTorch or TensorFlow.

The model is trained end-to-end and evaluated on held-out test data.

---

## Key Features
- Neural network implemented entirely from scratch (NumPy only)
- Custom implementations of:
  - One-hot encoding
  - ReLU activation and gradient
  - Softmax with numerical stability
  - Cross-entropy loss
  - Backpropagation
  - Stochastic Gradient Descent (SGD)
- Reproducible training and evaluation pipeline
- Clean, single-command execution

---

## Model Architecture
- **Input:** 784 features (28×28 flattened grayscale image)
- **Hidden Layer:** Fully connected layer with ReLU activation
- **Output:** 10-class softmax classifier (digits 0–9)


---

## Dataset
- **MNIST Handwritten Digits**
- Loaded using `scikit-learn`
- 60,000 training samples
- 10,000 test samples
- Pixel values normalized to `[0, 1]`
- Mean-centered using training data statistics

---

## Training Details
- Optimizer: Stochastic Gradient Descent (SGD)
- Loss Function: Cross-entropy
- Batch Size: 32
- Epochs: 15
- Learning Rate: 0.05
- Regularization: Optional L2 weight decay

During training, accuracy is periodically evaluated on a subset of the training data for monitoring.

---

## Results
- **Test Accuracy:** ~97% (may vary slightly due to random initialization)
- Check results directory for more information

---

## How to Run

1. Install dependencies
pip install -r requirements.txt

2. Run train.py
