
# Handwritten Digit Recognition

This project implements handwritten digit recognition using three types of models:
1. **Dense Neural Network (Dense NN)**.
2. **Convolutional Neural Network (CNN)**.
3. **PyTorch Lightning for streamlined training**.

## Dataset
The dataset used is **MNIST**, a collection of 70,000 grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels. It is split into:
- 60,000 training images
- 10,000 testing images

## Models Implemented

### Dense Neural Network (Dense NN)
- Fully connected layers with ReLU activations.
- Dropout layers for regularization.
- Cross-entropy loss for classification.

### Convolutional Neural Network (CNN)
- Two convolutional layers with ReLU activations and max pooling.
- Fully connected layer for output classification.
- Dropout for regularization.
- Cross-entropy loss for classification.

### PyTorch Lightning
- Streamlines training and evaluation using PyTorch Lightning's modular framework.

## Workflow (CRISP-DM Methodology)
1. **Business Understanding**: The goal is to classify handwritten digits effectively.
2. **Data Understanding**: Visualization and preprocessing of the MNIST dataset.
3. **Data Preparation**: Normalization and splitting into training, validation, and test datasets.
4. **Modeling**: Training and evaluation of Dense NN and CNN models.
5. **Evaluation**: Comparing performance metrics (loss and accuracy) of both models.
6. **Deployment**: Visualization of training history for both models.

## Training History Visualization
Both Dense NN and CNN models include plots for:
- Training and validation loss history.
- Training and validation accuracy history.

## Prerequisites
- Python 3.x
- PyTorch
- PyTorch Lightning
- Matplotlib
- torchvision

Install the dependencies using:
```bash
pip install torch torchvision pytorch-lightning matplotlib
```

## How to Run
1. Clone this repository.
2. Ensure all dependencies are installed.
3. Run the Python script:
```bash
python main.py
```
4. View the training and validation results in the output plots.

## Results
The models demonstrate their ability to classify digits with high accuracy. 
- Dense NN: Simpler architecture with moderate accuracy.
- CNN: More sophisticated and achieves higher accuracy.

## Author
This project is a demonstration of machine learning using PyTorch and PyTorch Lightning.

---
