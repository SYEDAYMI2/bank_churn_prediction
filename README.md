# Bank Customer Churn Prediction

This project predicts whether a bank customer will churn using machine learning models trained on historical customer data.

The goal is to compare multiple classification algorithms and identify the best-performing model based on F1-score and ROC-AUC, considering class imbalance.

---

## Dataset

- **Name:** Churn Modelling Dataset  
- **Source:** https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling  
- **Records:** 10,000 customers  
- **Target Variable:** `Exited` (1 = churned, 0 = retained)

---

## Feature Engineering

The following additional features were engineered to improve model performance:

- Age groups
- Balance categories
- Credit score categories
- Salary categories
- Balance per product
- Tenure-to-age ratio
- Engagement score
- Financial stability score
- High-value customer indicator

Categorical variables (`Geography`, `Gender`) were label-encoded.

---

## Models Implemented

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Support Vector Machine (SVM)
- Neural Network (MLPClassifier)
- Soft Voting Ensemble

---

## Results

**Best Model:** CatBoost  
- **F1-Score:** ~0.59  
- **ROC-AUC:** ~0.87  
- **Accuracy:** ~86.8%

CatBoost performed best overall, balancing precision and recall on an imbalanced churn dataset.

---

## How to Run

This notebook is intended to be run on **Kaggle**.

1. Create a new Kaggle notebook
2. Import `bank-churn-prediction.ipynb`
3. Add the dataset via **Add Input**
4. Run all cells

> Note: Ensemble cross-validation is disabled by default due to computational cost.

---

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- XGBoost, LightGBM, CatBoost
- Kaggle Notebook environment
