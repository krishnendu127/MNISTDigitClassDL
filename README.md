# MNIST Digit Classification

## Overview
This project focuses on classifying hand-written digits using machine learning techniques. It utilizes the famous MNIST dataset, containing 28x28 grayscale images of digits from 0 to 9. The goal is to build a neural network model that accurately predicts the digit represented by each image.

## Dataset
The dataset used for this project is the MNIST dataset, which is available in the Keras library. It consists of 60,000 training images and 10,000 testing images, with each image labeled with the corresponding digit (0 to 9).

## Features
- **Data Loading and Preprocessing**: The dataset is loaded from Keras and preprocessed by scaling the pixel values to a range of [0, 1].
- **Neural Network Architecture**: A neural network model is built with an input layer, two hidden layers with 50 neurons each using ReLU activation, and an output layer with 10 neurons using the softmax activation function.
- **Model Training**: The neural network is trained on the training data for 10 epochs using the Adam optimizer and sparse categorical cross-entropy loss.
- **Model Evaluation**: The trained model is evaluated on the test data, achieving an accuracy of approximately 97%.
- **Confusion Matrix**: A confusion matrix is generated to visualize the model's performance in classifying each digit.

## Dependencies
- NumPy
- Matplotlib
- Seaborn
- OpenCV (cv2)
- TensorFlow
- Keras

## Usage
1. Ensure you have the necessary dependencies installed.
2. Run the provided code in a Python environment such as Jupyter Notebook or Google Colab.
3. The code will load, preprocess, train, evaluate, and make predictions using the MNIST dataset.
4. You can modify the code, experiment with different neural network architectures or hyperparameters, and explore further improvements in classification accuracy.
