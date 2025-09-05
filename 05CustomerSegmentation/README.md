## Customer Segmentation using K-Means Clustering 🛍️
A practical implementation of unsupervised machine learning to segment customers based on their spending habits and income.

### Description
This project focuses on identifying distinct groups of customers within a dataset from a mall. By analyzing features like Annual Income and Spending Score, we can apply the K-Means clustering algorithm to create meaningful customer segments. These segments provide valuable insights for targeted marketing strategies, helping businesses to better understand and cater to their diverse customer base.

### Dataset
The dataset used is the "Mall Customer Segmentation Data" from Kaggle. It contains basic information about mall customers.

Source: Kaggle - Mall Customer Segmentation Data

Features Used:

Annual Income (k$): The annual income of the customer.

Spending Score (1-100): A score assigned by the mall based on customer behavior and spending nature.

### Methodology
The project follows a standard machine learning workflow:

Data Exploration (EDA): The dataset was loaded, inspected for missing values, and visualized to understand the relationships between variables. A scatter plot of Annual Income vs. Spending Score clearly showed potential clusters.

Feature Selection: The Annual Income (k$) and Spending Score (1-100) columns were selected as the basis for clustering.

Finding Optimal Clusters: The Elbow Method was used to determine the optimal number of clusters (K). By plotting the Within-Cluster Sum of Squares (WCSS) for a range of K values, the "elbow" of the graph was identified at K=5.

Model Training: A K-Means model was trained on the data with n_clusters=5. The model assigned each customer to one of the five identified clusters.

### Results: The 5 Customer Segments
The analysis successfully segmented customers into five distinct groups, each with clear characteristics:

Cluster 1 (Target/VIP): High income and high spending score. These are the most valuable customers and the primary target for premium offers.

Cluster 2 (Careful): High income but low spending score. They have purchasing power but are selective. Marketing should focus on quality and value.

Cluster 3 (Standard): Average income and average spending score. This represents the general customer base.

Cluster 4 (Careless): Low income but high spending score. Potentially younger customers who are highly influenced by trends.

Cluster 5 (Sensible): Low income and low spending score. These customers are budget-conscious.

### Technologies Used
Python 3

Jupyter Notebook

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For implementing the K-Means clustering algorithm.

Matplotlib & Seaborn: For data visualization.

