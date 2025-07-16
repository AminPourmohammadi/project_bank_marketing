# Bank Marketing Campaign Prediction

## 📌 Project Overview
This project predicts whether a customer will subscribe to a term deposit based on the **UCI Bank Marketing Dataset**. It demonstrates an **end-to-end machine learning pipeline** with feature engineering, hyperparameter tuning, and model evaluation.

---

## 📚 Notebooks
- `bank_marketing_original.ipynb`: Main version with full pipeline and to use for scoring before the marketing campaign.

---

## 📂 Dataset
- Source: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- Task: Predict the binary target variable `y` (yes/no subscription).
- Features: Customer demographics, previous marketing contacts, and call details.

---

## ✅ Workflow
1. **Data Loading & EDA**
   - Summary statistics, missing values, class imbalance.
2. **Feature Engineering**
   - Custom transformer for interaction features.
   - Handling `pdays`, creating binary indicators.
3. **Preprocessing**
   - One-hot encoding for categorical features.
   - Scaling for numeric features.
4. **Modeling**
   - Logistic Regression, Random Forest, XGBoost.
   - Voting Classifier for ensemble learning.
5. **Hyperparameter Tuning**
   - RandomizedSearchCV with cross-validation.
6. **Evaluation**
   - Metrics: F1-score, ROC AUC, Average Precision, **Top-K Precision** (business-oriented metric).
   - ROC Curves & Feature Importance visualization.

---

## 📊 Results
| Model       | F1   | ROC AUC | Avg Precision | Top 2% Precision |
|-------------|------|---------|---------------|------------------|
| Logistic    | 0.31 | 0.70    | 0.34          | 0.71             |
| RandomForest| 0.37 | 0.73    | 0.38          | 0.72             |
| XGBoost     | 0.39 | 0.73    | 0.38          | 0.78             |
| Voting      | 0.38 | 0.72    | 0.38          | 0.74             |


*Top-K precision indicates precision for the top 2% of predicted customers—useful for targeted marketing.*

---

## 🛠 Requirements:

To install the dependencies, create a `requirements.txt` file with the following packages:
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
Then install:
pip install -r requirements.txt

