## Flower Species Classification using Transfer Learning 🌸
This project demonstrates the power and efficiency of Transfer Learning to build a highly accurate image classifier. A pre-trained Convolutional Neural Network (VGG16) is adapted to classify images of flowers into five distinct species.

The model successfully leverages the pre-trained features to achieve high accuracy (~81.23%) with a relatively small dataset and minimal training time.

### Overview
The goal of this project is to classify images of flowers into one of five categories: daisy, dandelion, rose, sunflower, or tulip.

Instead of building and training a new Convolutional Neural Network from scratch—a process that would require a massive dataset and extensive computation—this project employs Transfer Learning. We use the VGG16 model, which was pre-trained on the vast ImageNet dataset, as a powerful feature extractor. By freezing the pre-trained layers and adding a new, custom classifier on top, we can quickly create a highly effective model tailored to our specific task.

This approach highlights a key technique used in modern computer vision, demonstrating how to achieve state-of-the-art results efficiently.

### Dataset
The project uses the "Flowers Recognition" dataset, which contains over 4,000 images of flowers.

Source: Kaggle - Flowers Recognition Dataset

Classes: The dataset is organized into 5 subdirectories, each corresponding to a class:

daisy

dandelion

rose

sunflower

tulip

Structure: This directory structure is ideal for use with the tf.keras.utils.image_dataset_from_directory utility, which automatically infers class labels from the folder names.

### Project Workflow
The core of this project lies in the implementation of the transfer learning workflow.

1. Data Loading and Preparation
Images were loaded from their directories using image_dataset_from_directory.

All images were resized to 224x224 pixels to match the required input shape for the VGG16 model.

The data was automatically batched and split into an 80% training set and a 20% validation set.

2. Transfer Learning Implementation
Base Model: The VGG16 model was loaded from tf.keras.applications with weights pre-trained on ImageNet.

Removing the Top Layer: The model was loaded with include_top=False. This crucial step removes the original final classification layer, which was trained to classify 1000 ImageNet categories.

Freezing the Base: All layers of the VGG16 base model were frozen by setting base_model.trainable = False. This locks in the powerful, generic visual features (edges, textures, patterns) that the model has already learned.

3. Custom Model Architecture
A new Sequential model was built by stacking a custom classifier on top of the frozen VGG16 base.

The frozen base_model (VGG16) as the first layer.

A Flatten layer to convert the feature maps from the base model into a 1D vector.

A Dense hidden layer with 256 neurons and ReLU activation to learn combinations of features specific to flowers.

A final Dense output layer with 5 neurons (one for each flower class) and a softmax activation function to produce a probability distribution.

4. Training and Evaluation
The model was compiled with the Adam optimizer and sparse_categorical_crossentropy loss function.

Training was performed for 10 epochs. Thanks to transfer learning, the model achieved high validation accuracy very quickly.

### Results
The model demonstrated excellent performance, validating the effectiveness of the transfer learning approach.

Validation Accuracy: The final model achieved a validation accuracy of approximately 81.23%.

The training history shows a rapid convergence to a high accuracy with low loss, highlighting the efficiency of leveraging pre-trained weights.

### Technologies Used
Python

TensorFlow and its Keras API

tf.keras.applications for loading pre-trained models (VGG16).

tf.keras.utils.image_dataset_from_directory for efficient data loading.

NumPy

Matplotlib for data visualization

Jupyter Notebook

### How to Run
Clone the repository:

git clone https://github.com/Kajalmehta29/LearnAI.git

cd LearnAI/08FlowerRecognition

Download the Dataset: Download the "Flowers Recognition" dataset from the Kaggle link above and unzip it into the project folder. Ensure you have the flowers directory with its 5 subdirectories.

###Create and activate a virtual environment:

It is recommended to use a compatible Python version (e.g., 3.11).

python3.11 -m venv myenv

source myenv/bin/activate

###Install the required libraries:

pip install -r requirements.txt

###Launch Jupyter Notebook and run the code:

jupyter notebook

Open the .ipynb file and run the cells sequentially.