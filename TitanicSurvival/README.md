# Project 2: Titanic Survival Prediction 🚢

## Project Goal
The goal of this project is to build a machine learning model that predicts whether a passenger on the Titanic survived or not. This is a classic binary classification problem, and a great exercise for practicing data cleaning, feature engineering, and model training.

## Dataset
The data is from the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition on Kaggle. It contains passenger information like age, gender, ticket class, and whether they survived.

## Workflow
1.  **Data Loading & Exploration:** The `train.csv` dataset was loaded into a Pandas DataFrame. I performed an initial Exploratory Data Analysis (EDA) to understand the features and identify missing values in columns like `Age`, `Cabin`, and `Embarked`.

2.  **Data Cleaning:**
    * Missing `Age` values were filled with the median age.
    * The `Cabin` column was dropped due to a high number of missing values.
    * Missing `Embarked` values were filled with the most frequent port.

3.  **Feature Engineering:**
    * The categorical `Sex` and `Embarked` columns were converted into numerical format using `pd.get_dummies()`. This is necessary for the model to process them.
    * Unnecessary columns like `Name`, `Ticket`, and `PassengerId` were dropped as they don't provide generalizable patterns for a simple model.

4.  **Model Training:**
    * The data was split into a training set (80%) and a testing set (20%).
    * A **Logistic Regression** model was trained on the prepared data. This model is well-suited for binary classification tasks.

5.  **Evaluation:**
    * The model's performance was evaluated on the unseen test data.
    * The key metric used was **Accuracy**, which measures the percentage of correct predictions.

## Results
The model achieved an **accuracy of 81.01%** on the test set. The confusion matrix below shows the breakdown of correct and incorrect predictions:

|            | Predicted: No | Predicted: Yes |
|------------|---------------|----------------|
| **Actual: No** |      90     |       15       |
| **Actual: Yes**|      19     |       55       |

This result is a strong baseline and demonstrates a solid understanding of the classification workflow.

## Libraries Used
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn