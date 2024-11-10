# Breast-Cancer-Machine-learning
## 🩺 Breast Cancer Prediction with Logistic Regression

Hey there! 👋 Welcome to my project on predicting breast cancer diagnoses with logistic regression. This project was born out of my journey into machine learning and deep learning, and I’m excited to share it with you. Using the Wisconsin Breast Cancer dataset, I built a model that can classify tumors as benign or malignant. 

## 📊 Project Overview

Breast cancer is one of the most common cancers globally, and early detection is crucial for effective treatment. With machine learning, we can build models that assist in identifying cases needing further investigation. Here, I’m using logistic regression to create a model that’s both effective and easy to interpret.

## 🔍 Workflow

1. Data Exploration: After loading the data, I examined the relationships between features and identified any patterns. I also visualized the correlation between features to highlight the strongest relationships.
   
2. Feature Selection: I reduced redundancy by removing features with a correlation above 0.9. This step helps streamline the model and prevent overfitting.

3. Data Preprocessing: I split the data into training and test sets and standardized the features. Scaling ensures that each feature contributes equally to the model's decision-making.

4. Model Training: I trained a logistic regression model and evaluated it with accuracy, precision, recall, and F1 scores to understand its performance.

5. Evaluation: Finally, I assessed the model's accuracy using a confusion matrix and classification report to see how well it distinguishes between benign and malignant tumors.
