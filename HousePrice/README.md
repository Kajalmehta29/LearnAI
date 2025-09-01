# California House Price Prediction 🏡

## Project Goal
The goal of this project is to predict the median house value in California districts using a linear regression model.

## Dataset
The dataset used is the California Housing Prices dataset from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

## Workflow
1.  **Data Loading:** Loaded the `housing.csv` file using Pandas.
2.  **Data Cleaning:** Filled missing values in the `total_bedrooms` column with the median.
3.  **Exploratory Data Analysis (EDA):** Visualized the distribution of house prices.
4.  **Model Training:** Trained a Linear Regression model on the numerical features.
5.  **Evaluation:** The model was evaluated using Root Mean Squared Error (RMSE).

## Results
The baseline Linear Regression model achieved an **RMSE of $71,133.17**.


## How to Run
1.  Clone the repository.
2.  Install the required libraries: `pip install -r requirements.txt`
3.  Run the `House_Price_Prediction.ipynb` Jupyter Notebook.