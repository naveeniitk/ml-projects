# Telco Customer Churn Prediction

This project predicts customer churn for a telecommunications company using supervised machine learning models. The objective is to identify customers likely to stop using the service based on their demographics and usage behavior.

---

## Dataset

- **Source**: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target Variable**: `Churn` (Yes/No)

---

## Features

The dataset contains both categorical and numerical features:

- **Categorical**: `gender`, `Partner`, `Dependents`, `PhoneService`, `InternetService`, `Contract`, `PaymentMethod`, etc.
- **Numerical**: `tenure`, `MonthlyCharges`, `TotalCharges`

---

## Preprocessing

1. Dropped unnecessary identifiers (e.g., `customerID`)
2. Converted `TotalCharges` to numeric (handled missing/blank values)
3. Encoded binary columns (`Yes/No`, `True/False`) as 1/0
4. Applied one-hot encoding to categorical variables with multiple classes
5. Standardized data formats and types

---

## Models Used

### Logistic Regression

- Baseline binary classification model
- Interpretable coefficients

### Decision Tree Classifier

- Simple tree-based model
- Visualized for interpretability

### Random Forest Classifier

- Ensemble of decision trees
- Higher performance and reduced overfitting

---

## Evaluation Metrics

- Accuracy Score
- Predictions compared to actual churn labels

---

## Visualizations

- Decision tree plotted using `sklearn.tree.plot_tree`
- Individual estimator from the random forest visualized for clarity

---

## Results Summary

| Model               | Accuracy (approx.) |
| ------------------- | ------------------ |
| Logistic Regression | 0.80+              |
| Decision Tree       | 0.78+              |
| Random Forest       | 0.81+              |

---
