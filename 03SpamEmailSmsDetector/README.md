# Project 3: Spam Message Detector 📧

## Project Goal
This project is an introduction to Natural Language Processing (NLP). The objective is to build a machine learning model capable of classifying text messages as either "spam" or "ham" (not spam).

## Dataset
The project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from the UCI Machine Learning Repository, available on Kaggle. The dataset contains a collection of 5,572 SMS messages tagged with their respective labels.

## Workflow
1.  **Data Loading and Preprocessing:**
    * The dataset was loaded using Pandas.
    * Unnecessary columns were dropped, and the remaining columns were renamed to `label` and `message`.
    * The categorical `label` column ('ham', 'spam') was converted into a numerical format (0, 1).

2.  **Text Vectorization:**
    * To prepare the text data for the model, I used **`CountVectorizer`** from Scikit-learn.
    * This technique converts text messages into a matrix of token counts, which is a numerical representation that the model can understand.

3.  **Model Training:**
    * The vectorized data was split into training (80%) and testing (20%) sets.
    * A **Multinomial Naive Bayes** classifier was chosen for this task. This algorithm is highly efficient and performs exceptionally well on text classification problems.

4.  **Evaluation:**
    * The model was evaluated on the test set using **Accuracy** and a **Confusion Matrix**.
    * A final step was added to test the trained model on new, custom-written messages to see it perform in a real-world scenario.

## Results
The model achieved an outstanding **accuracy of 97.85%**. This demonstrates the effectiveness of the Naive Bayes algorithm for text-based classification tasks. The model was also successful in correctly classifying new, unseen messages as either spam or ham.

## Libraries Used
* Pandas
* Scikit-learn (`CountVectorizer`, `MultinomialNB`, `train_test_split`, `accuracy_score`)