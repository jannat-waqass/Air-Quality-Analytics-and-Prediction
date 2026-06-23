Air Quality Analytics and Prediction

#Overview

This project focuses on analyzing and predicting air quality levels using machine learning techniques. By integrating multiple air pollution datasets, we developed and evaluated several classification models to determine the most effective approach for predicting air quality categories. The project also includes data preprocessing, exploratory data analysis (EDA), clustering, and model comparison.

#Features
Integration of multiple air quality datasets
Data cleaning and preprocessing
Exploratory Data Analysis (EDA)
K-Means Clustering for pattern discovery
Classification using:
Random Forest
Decision Tree
K-Nearest Neighbors (KNN)
Naïve Bayes
Performance evaluation using:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Comparative analysis of machine learning models
Dataset

The project utilizes two publicly available air quality datasets:

Global Air Pollution Dataset
Air Quality and Pollution Assessment Dataset

The datasets were merged and preprocessed to create a comprehensive dataset for training and evaluation.

#Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook

#Methodology
1. Data Preprocessing
Handling missing values
Removing duplicates
Data integration
Feature selection
Data transformation and normalization
2. Exploratory Data Analysis
Distribution analysis
Correlation analysis
Visualization of key environmental factors
3. Clustering
K-Means Clustering to identify patterns and similarities within air quality data
4. Machine Learning Models

The following models were trained and evaluated:

Random Forest
Decision Tree
K-Nearest Neighbors (KNN)
Naïve Bayes
Results

The project demonstrated strong predictive performance across multiple models.

#Model	Accuracy
Random Forest	97%
Decision Tree	96%
KNN	87%
Naïve Bayes	Lower performance due to feature independence assumptions

Random Forest achieved the best overall performance, particularly for minority air quality classes, making it the preferred model for this project.

#Key Findings
Random Forest provided the highest overall accuracy and robustness.
Decision Tree achieved competitive performance with interpretable decision rules.
KNN performed adequately but struggled with imbalanced classes.
Naïve Bayes was less effective due to dependencies among environmental features.
Data quality and preprocessing significantly impacted model performance.
