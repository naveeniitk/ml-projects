## Iris Species Classification and Visualization
This project explores the classic Iris dataset using a variety of data visualizations and machine learning models to classify iris flower species based on sepal and petal measurements.

## Linear Regression from Scratch

This repository demonstrates how to implement Linear Regression from scratch in Python using **NumPy**, **Pandas**, and **Matplotlib**, along with comparisons to **scikit-learn**'s implementation.

It includes two approaches:

#### 1. Gradient Descent

An iterative optimization technique that adjusts weights to minimize the Mean Squared Error (MSE). Useful for large datasets and online learning scenarios. Training progress is visualized through MSE vs. Iterations.

#### 2. Closed-Form Solution (Normal Equation)

A mathematical one-step solution derived from linear algebra. The formula used is:

**θ = (XᵀX)⁻¹ Xᵀ y**

Where:
- `X` is the input feature matrix (with an added bias column),
- `y` is the target vector,
- `θ` is the resulting parameter vector (weights + bias).

This method computes exact weights without iterations and is suitable for smaller datasets where matrix inversion is computationally feasible.

Both implementations are validated against **Scikit-learn's** `LinearRegression` for correctness.
