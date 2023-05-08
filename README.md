# Particle Drag Detection Model

The Jupyter Notebook provided contains code for importing necessary libraries, data preprocessing, model creation, training, and evaluation. This model is used to detect particle drag in Selective Laser Sintering processes.

## Requirements

To run the provided Jupyter Notebook, you will need the following libraries:

- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- TensorFlow Datasets
- Pandas

## Usage

1. Place your dataset in the `data` directory. The dataset should contain two subdirectories: `particle_drag` and `ok`. Each subdirectory should contain images of the corresponding class.
2. Run the Jupyter Notebook. It will perform the following steps:
    - Import necessary libraries
    - Remove corrupt images from the dataset
    - Load data and preprocess it
    - Split data into training, validation, and testing sets
    - Define the model architecture
    - Compile the model
    - Train the model
    - Evaluate the model performance
    - Save the trained model

After training, the model will be saved in the `models` directory.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

- Conv2D (32 filters, kernel size 3x3, activation 'relu')
- MaxPooling2D
- Conv2D (64 filters, kernel size 3x3, activation 'relu')
- MaxPooling2D
- Conv2D (64 filters, kernel size 3x3, activation 'relu')
- MaxPooling2D
- Flatten
- Dense (256 neurons, activation 'relu')
- Dense (2 neurons, activation 'softmax')

The model is compiled with the Adam optimizer, Binary Crossentropy loss, and accuracy metric.

## Evaluation Metrics

The model performance is evaluated using the following metrics:

- Precision
- Recall
- Binary Accuracy

These metrics are calculated on the test dataset after training the model.