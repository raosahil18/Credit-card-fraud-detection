# FindDefault
## Prediction of Credit Card Fraud (Classification)
#### Data source : creditcard.csv or the file is also uploaded on github.
## Overview
Credit card fraud is a wide-ranging term for theft and fraud committed using or involving a payment card, such as a credit card or debit card, as a fraudulent source of funds in a transaction. The purpose may be to obtain goods without paying or to obtain unauthorized funds from an account. Credit card fraud is also an adjunct to identity theft. Although incidences of credit card fraud are limited to about 0.1% of all card transactions, they have resulted in huge financial losses as the fraudulent transactions have been large value transactions. It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. What we need is an algorithm, which could classify a transaction as fraudulent or non-fraudulent. Doing so will benefit both the credit card companies and the customers who have to go through the ordeal.

### Introduction:

### Dataset:
The dataset used in this project contains transactions made by credit cards in September 2013 by European cardholders. It includes a total of 284,807 transactions, out of which 492 are labeled as fraudulent. The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.

Due to confidentiality issues, there are not provided the original features of the data.

1. Features V1, V2, ... V28 are the principal components obtained with PCA;
2. The only features which have not been transformed with PCA are Time and Amount. Feature Time contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature Amount is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.
3. Feature Class is the response variable and it takes value 1 in case of fraud and 0 for genuine transaction.


### Project Goals:
1. Explore and analyze the dataset to identify patterns and relationships.
2. Preprocess the data by handling missing values, outliers and imbalanced classes.
3. Exploratory Data Analysis (EDA).
4. Train machine learning models to predict fraudulent transactions.
5. Hyperparameter Tuning/Model Improvement 
6. Evaluate the performance of the models using appropriate metrics.

### Technologies Used:
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Oversampling (SMOTE)

### Model Used
- Logistic Regression
- Decision Tree
- XGBoost

### Project Structure:	
The repository contains the following files:
- `notebooks`: Directory containing Jupyter Notebook files or Python scripts used for training the machine learning models. 
- `Model deployment plan`: This document contains the deployment plan for the model of this project.
- `README.md`: This document offers a comprehensive project overview , project goals , future work etc.

### Future Work:
1.Gather a more comprehensive dataset over an extended duration to encompass a wider spectrum of fraudulent behaviors.
2. Explore different  approaches for managing imbalanced datasets, visualizations etc.
3.Utilize advanced methodologies for streaming data and dynamic feature engineering to bolster the effectiveness of fraud detection systems in real-time.
4. Consistently observe and refine the deployed model to accommodate changing fraud patterns.

## Model Deployment plan
- There is the deployment plan for the model of this project.
