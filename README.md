# Cuisine-Classification
Objective: Develop a machine learning model to classify restaurants based on their cuisines.

---

This project predicts restaurant ratings based on various features like location, average cost, price range, votes, and encoded attributes like restaurant name, rating colour, and rating text. The goal is to identify factors influencing restaurant ratings and build a robust machine-learning model to predict ratings effectively.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Analysis of Model Bias and Challenges](#analysis-of-model-bias-and-challenges)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project uses machine learning to predict the ratings of restaurants based on various features. The dataset includes multiple attributes, such as cost, location coordinates, and various categorical descriptors of the restaurant. The core objective is to preprocess the data, build a classification model, evaluate its performance, and analyze results across different cuisine types to identify any biases or challenges in the model's predictions.

## Dataset

The dataset includes information on various attributes of restaurants:
- **Restaurant ID**: Unique identifier for each restaurant
- **Location Attributes**: Country Code, Longitude, Latitude
- **Cost Attributes**: Average Cost for Two, Price Range
- **Feedback Attributes**: Votes, Rating Color, Rating Text
- **Encoded Features**: Restaurant Name, Rating Categories, Cuisines (one-hot encoded)

### Missing Values and Categorical Encoding
- Missing values were handled using median imputation for numeric columns and mode imputation for categorical columns.
- Categorical variables were one-hot encoded, expanding each category into binary features.

---

## Project Structure

```plaintext
├── data/               # Folder containing the dataset file
├── notebooks/          # Jupyter notebooks for code execution
├── src/                # Source code files for the project
│   ├── preprocessing.py    # Scripts for data cleaning and preprocessing
│   ├── train_model.py      # Code for model training and evaluation
│   ├── evaluate_model.py   # Performance evaluation and metric calculation
├── README.md           # Project documentation
└── requirements.txt    # List of required packages
```

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/restaurant-rating-prediction.git
   cd restaurant-rating-prediction
   ```

2. **Install Dependencies**:

   Install the necessary packages using `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Colab** (Optional):

   If you’re using Google Colab, upload the dataset manually or mount Google Drive to access it.

---

## Data Preprocessing

The dataset undergoes several preprocessing steps:
1. **Missing Value Handling**:
   - Numeric columns: Filled missing values with the median.
   - Categorical columns: Filled missing values with the most frequent category (mode).

2. **Categorical Encoding**:
   - Categorical features such as restaurant names, rating color, and rating text were one-hot encoded.

3. **Feature Engineering**:
   - Created new columns based on one-hot encoded cuisines for analyzing performance across cuisine types.

4. **Data Splitting**:
   - Split the data into training and testing sets (80% training, 20% testing).

---

## Exploratory Data Analysis

We performed basic EDA, including:
- Distribution plots for numeric features like price range and votes.
- Bar charts for categorical variables like rating text and rating color.
- Analysis of correlations between numeric features.

---

## Model Training

A Random Forest Classifier was selected as the classification algorithm for this project. Key steps included:

1. **Splitting the Data**:
   - Training and testing sets were prepared, with the model trained on the training set.

2. **Training**:
   - We fit the Random Forest Classifier on the training data.

3. **Parameter Tuning**:
   - Basic hyperparameter tuning was performed to improve model performance.

---

## Model Evaluation

We evaluated the model’s performance using metrics such as:
- **Accuracy**: Percentage of correct predictions over the total predictions.
- **Precision**: Ratio of correct positive predictions to total positive predictions.
- **Recall**: Ratio of correctly predicted positives to all actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

A confusion matrix and a bar chart of evaluation metrics were also generated for visualization.

### Code Snippet for Evaluation:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Evaluate the model on the test data
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, average='weighted')
test_recall = recall_score(y_test, test_predictions, average='weighted')
test_f1 = f1_score(y_test, test_predictions, average='weighted')

# Display metrics
print("Testing Accuracy:", test_accuracy)
print("Testing Precision:", test_precision)
print("Testing Recall:", test_recall)
print("Testing F1 Score:", test_f1)
print("\nClassification Report:\n", classification_report(y_test, test_predictions))
```

---

## Analysis of Model Bias and Challenges

We analyzed the model's performance across different cuisines by:
1. Grouping test data based on one-hot encoded cuisine types.
2. Calculating performance metrics (accuracy, precision, recall, F1-score) for each cuisine.
3. Identifying potential biases or challenges in the model, such as:
   - Lower accuracy or F1-scores for specific cuisine types.
   - Skewed performance metrics indicating data imbalance or underrepresented cuisines.

---

## Results

- **Overall Model Performance**:
  The model achieved satisfactory accuracy, precision, recall, and F1-score on the test data, demonstrating a robust ability to predict restaurant ratings.

- **Cuisine-based Performance**:
  Performance varied across cuisines, with certain cuisines achieving lower F1-scores. This suggests potential bias, possibly due to class imbalance or distinctive cuisine features not well-represented in the model.

---

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Submit a pull request.

---

## License

This project is licensed under the MIT License.

--- 

## Contact

For more information, feel free to contact the project maintainer:  
**Name:** Nsidibe Daniel Essang  
**Email:** nsidibedaniel62@gmail.com  
**LinkedIn:** [Nsidibe Essang](https://www.linkedin.com/in/nsidibe-essang-142778204/)

---
