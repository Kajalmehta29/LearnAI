## Movie Review Sentiment Analysis using NLP 🎬
This project is an end-to-end implementation of a machine learning model that classifies movie reviews from the IMDB dataset as either positive or negative. It leverages Natural Language Processing (NLP) techniques to process and understand the text data.

### Description
The goal of this project is to build an effective sentiment analysis model. By training on a large dataset of 50,000 movie reviews, the model learns to associate specific words and phrases with positive or negative sentiment. This has practical applications in areas like customer feedback analysis, social media monitoring, and product review evaluation.

The final model achieves an accuracy of approximately 89.14% on unseen data and can be used to predict the sentiment of new, custom-written reviews.

### Dataset
The dataset used is the IMDB Dataset of 50K Movie Reviews, which is a classic benchmark for sentiment analysis tasks.

Source: Kaggle - IMDB Dataset of 50K Movie Reviews

Content: The dataset contains 50,000 reviews, evenly split with 25,000 positive and 25,000 negative entries. This balance is ideal for training a classification model.

### Project Workflow
The project follows a systematic approach to cleaning the data, training a model, and evaluating its performance.

1. Data Loading and Exploration
The data was loaded into a Pandas DataFrame.

Initial exploration confirmed the dataset's structure and the balanced distribution of positive and negative sentiments.

2. Text Preprocessing 🧹
To prepare the text for the model, a comprehensive cleaning pipeline was implemented:

Removed HTML tags (e.g., <br />) using regular expressions.

Removed punctuation, numbers, and special characters to ensure only alphabetical text remained.

Converted all text to lowercase for uniformity.

Removed common English "stopwords" (e.g., 'the', 'a', 'is') using the NLTK library to help the model focus on meaningful words.

3. Feature Engineering (Text Vectorization)
Computers don't understand words, so the cleaned text was converted into a numerical format using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.

The TfidfVectorizer from Scikit-learn was used for this task.

The vocabulary was limited to the top 5,000 most frequent words to maintain efficiency and reduce noise.

4. Model Training
The dataset was split into an 80% training set and a 20% testing set.

A Logistic Regression model was chosen for its efficiency and strong performance on text classification tasks.

The model was trained on the TF-IDF vectors of the training data.

### Results and Evaluation 📊
The model's performance was evaluated on the unseen test data, yielding strong results:

Accuracy: The model achieved an overall accuracy of 89.14%.

The detailed classification report shows excellent performance for both classes:

               precision    recall  f1-score   support

    negative       0.90      0.88      0.89      4961
    positive       0.88      0.91      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
The model also proved effective at classifying new, custom-written reviews, demonstrating its ability to generalize.

### Technologies Used
Python 3

Jupyter Notebook

Pandas: For data manipulation and analysis.

NLTK (Natural Language Toolkit): For stopword removal.

Scikit-learn: For data splitting, TF-IDF vectorization, model training (LogisticRegression), and evaluation.
