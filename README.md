# MNIST Image Classification with Convolutional Neural Networks

This project implements an image classification model using the MNIST dataset and a convolutional neural network (CNN) built with TensorFlow and Keras.

## Overview

The model classifies handwritten digits from the MNIST dataset into 10 classes (digits 0-9). It uses convolutional layers, max pooling, and fully connected layers to extract features and perform the classification.

## Features

- Loads and preprocesses the MNIST dataset
- Converts grayscale images to RGB format and resizes them to `64x64`
- Implements a CNN with three convolutional blocks followed by fully connected layers
- Compiles the model with the Adam optimizer and categorical cross-entropy loss
- Trains the model on the MNIST training set
- Predicts the class of digits from the test set

## Dependencies

- Python (3.x)
- TensorFlow
- NumPy

## Installation

1. Clone this repository:

   git clone https://github.com/your-username/your-repository.git
   
   cd your-repository

## How It Works

**Load Dataset**: Downloads and loads the MNIST dataset.

# Preprocess Data:

  . Normalizes pixel values to the range [0, 1].

  . Resizes images from 28x28 to 64x64 to match the CNN input shape.

  . Converts single-channel grayscale images to three-channel RGB images.

# Build the Model:

  . Adds three convolutional blocks with increasing filter sizes (32, 64, and 128).

  . Applies max pooling after each convolutional layer.

  . Flattens the output and uses a dense layer with 128 units followed by a dropout layer.

  . Outputs predictions through a softmax-activated dense layer.

  . Train the Model: Uses the Adam optimizer and trains the model for 5 epochs.

  . Make Predictions: Predicts the class of test images.

# Model Architecture

  . Layer Type	Output Shape	Parameters

  . Conv2D (32 filters)	(62, 62, 32)	896

  . MaxPooling2D	(31, 31, 32)	0

  . Conv2D (64 filters)	(29, 29, 64)	18,496

  . MaxPooling2D	(14, 14, 64)	0

  . Conv2D (128 filters)	(12, 12, 128)	73,856

  . MaxPooling2D	(6, 6, 128)	0

  . Flatten	(4608)	0

  . Dense (128 units)	(128)	589,952

  . Dropout (rate=0.5)	(128)	0

  . Dense (10 classes)	(10)	1,290

  . Total parameters: 684,490


## Usage

Run the script:

python mnist_cnn.py

**Output**:

  . Displays training progress and accuracy per epoch.

  . Outputs the predicted class for the first test image.

# Results

**Accuracy**: Achieved an accuracy of 99% on the training set after 5 epochs.

# Future Improvements

  . Experiment with more advanced architectures (e.g., ResNet, VGG).

  . Fine-tune hyperparameters such as learning rate, dropout rate, and batch size.

  . Evaluate the model on additional datasets for generalization.

## Acknowledgments

TensorFlow and Keras documentation

MNIST Dataset
