# Project 5: Iris Flower Classification 🌸

## Project Goal
This project focuses on multi-class classification, a fundamental machine learning task. The objective is to build a model that can classify Iris flowers into one of three distinct species (Setosa, Versicolor, or Virginica) based on four physical measurements: sepal length, sepal width, petal length, and petal width.

## Dataset
The project utilizes the classic **Iris flower dataset**, which is conveniently included with the Scikit-learn library. The dataset is small, clean, and contains 150 samples, with 50 samples for each of the three species.

## Workflow
1.  **Data Loading:** The Iris dataset was loaded directly from `sklearn.datasets`.
2.  **Exploratory Data Analysis (EDA):** The data was converted into a Pandas DataFrame for easy manipulation. A **Seaborn `pairplot`** was used to create a grid of scatterplots, which visually demonstrated the clear separation between the species based on their features.
3.  **Model Training:**
    * The data was split into a training (70%) and testing (30%) set.
    * A **K-Nearest Neighbors (KNN)** classifier was chosen for this task. KNN is an intuitive algorithm that classifies new data points based on the majority class of their 'k' nearest neighbors.
4.  **Evaluation:** The model's performance was assessed using **Accuracy** and a **Confusion Matrix**.

## Results
The KNN model achieved an outstanding **accuracy of 100%** on the test set, correctly classifying all unseen samples. This high performance is expected due to the distinct, linearly separable nature of the Iris dataset.

## Concepts Learned
* Multi-class classification.
* Using built-in Scikit-learn datasets.
* The K-Nearest Neighbors (KNN) algorithm.
* Using `seaborn.pairplot` for effective EDA.