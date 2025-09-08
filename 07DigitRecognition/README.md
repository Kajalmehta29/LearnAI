## Handwritten Digit Recognition using a Neural Network ✍️
This project is an introduction to deep learning and computer vision, demonstrating how to build a neural network to classify handwritten digits from the famous MNIST dataset. The model is built using TensorFlow and its high-level Keras API.

The final trained model successfully achieves an accuracy of over 97% on the unseen test data.

### Overview
The goal of this project is to create a model that can correctly identify a digit (0-9) from a 28x28 pixel grayscale image. This is a classic "Hello, World!" problem in the field of deep learning, providing a practical foundation for understanding the core concepts of neural networks, including layers, activation functions, loss functions, and optimizers.

### Dataset
The project utilizes the MNIST (Modified National Institute of Standards and Technology) dataset, which is a benchmark dataset in the computer vision community.

Content: It consists of 70,000 images of handwritten digits.

60,000 images for the training set.

10,000 images for the testing set.

Format: Each image is a 28x28 pixel grayscale image, with pixel values ranging from 0 to 255.

Loading: The dataset is conveniently included with the tensorflow.keras.datasets library, making it easy to load and use.

### Project Workflow
The project follows a standard machine learning pipeline from data preparation to model evaluation.

1. Data Loading and Preprocessing
The MNIST dataset was loaded directly from keras.datasets.mnist.

Normalization: The pixel values of the images were scaled from their original range of [0, 255] to a new range of [0, 1] by dividing them by 255.0. This step is crucial for helping the neural network train efficiently.

2. Model Architecture
A simple but effective neural network was constructed using the Keras Sequential API. The architecture is as follows:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 dense (Dense)               (None, 128)               100480

 dense_1 (Dense)             (None, 10)                1290
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
Input Layer: A Flatten layer that transforms each 28x28 image matrix into a 1D vector of 784 pixels.

Hidden Layer: A Dense layer with 128 neurons and a ReLU (Rectified Linear Unit) activation function. This is the main processing layer where the model learns complex patterns.

Output Layer: A final Dense layer with 10 neurons (one for each digit class from 0-9) and a Softmax activation function. The softmax function converts the layer's output into a probability distribution, giving the model's confidence for each digit.

3. Compilation and Training
The model was compiled using the Adam optimizer, a highly effective and popular optimization algorithm.

The sparse_categorical_crossentropy loss function was used to measure the model's error during training.

The model was trained for 10 epochs, meaning it processed the entire training dataset 10 times.

### Results
The model's performance was evaluated on the 10,000 unseen images in the test set.

Test Accuracy: The model achieved a final accuracy of ~97.8%, demonstrating its effectiveness and ability to generalize to new data.

### Technologies Used
Python

TensorFlow and its Keras API

NumPy for numerical operations

Matplotlib for visualizing the data

Jupyter Notebook for development

### How to Run
Clone the repository:

Bash

git clone https://github.com/Kajalmehta29/LearnAI.git
cd LearnAI/07DigitRecognition
Create and activate a virtual environment:
It is recommended to use a compatible Python version (e.g., 3.11) for TensorFlow on macOS.

Bash

# Create the environment with a specific Python version
python3.11 -m venv myenv

# Activate the environment
source myenv/bin/activate
Install the required libraries:

Bash

pip install --upgrade pip
pip install tensorflow-macos numpy matplotlib jupyter
pip install tensorflow-metal
Launch Jupyter Notebook and run the code:

Bash

jupyter notebook
Open the .ipynb file and run the cells sequentially to train and evaluate the model.