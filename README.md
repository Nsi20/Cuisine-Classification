# Cuisine-Classification
Objective: Develop a machine learning model to classify restaurants based on their cuisines.

## Overview

This project aims to develop a machine learning model to classify restaurants based on their cuisines using a dataset of restaurant information. The model uses features like restaurant location, price range, and customer ratings to predict the cuisine category.

## Dataset

The project uses a CSV file named `Dataset (1).csv` containing restaurant data. The dataset includes features such as restaurant name, location, cuisines, average cost, aggregate rating, and more.

## Methodology

1. **Data Preprocessing:**
   - Handling missing values by imputing numerical features with their mean and categorical features with 'Unknown'.
   - Encoding categorical variables using one-hot encoding.

2. **Data Splitting:**
   - Splitting the data into training and testing sets (80% training, 20% testing) using `train_test_split`.

3. **Model Selection and Training:**
   - Converting 'Aggregate rating' into categorical labels (Low, Medium, High).
   - Using a Random Forest Classifier for cuisine classification.
   - Training the model on the training data.

4. **Model Evaluation:**
   - Evaluating the model's performance on the testing data using accuracy, precision, recall, and F1-score metrics.
   - Analyzing the model's performance across cuisines to identify potential challenges or biases.

## Results

The model achieved a high accuracy on the testing data (around 95%), indicating its effectiveness in classifying cuisines. The performance across different cuisines was also analyzed using F1-score, revealing potential areas for improvement.

## Usage

1. Upload the `Dataset (1).csv` file to your Google Colab environment.
2. Run the provided Python code in the notebook.
3. The code will preprocess the data, train the model, and evaluate its performance.
4. You can modify the code to experiment with different classification algorithms or hyperparameters.

## Dependencies

- pandas
- scikit-learn
- matplotlib

## Future Work

- Explore other classification algorithms and compare their performance.
- Fine-tune the model's hyperparameters to further improve accuracy.
- Address potential biases and improve performance for specific cuisines.
- Develop a user interface for interacting with the model.
