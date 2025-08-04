# Linear Regression

This repository demonstrates how to implement Linear Regression from scratch in Python using **NumPy**, **Pandas**, and **Matplotlib**, along with comparisons to **scikit-learn**'s implementation.

It covers:

- Linear Regression using **Gradient Descent**
- Linear Regression using the **Closed-Form Solution (Normal Equation)**
- Visualization of training performance via **MSE vs Iterations**
- Comparison with **Scikit-learn's LinearRegression**

---

## Linear Regression (Closed-Form using Normal Equation)

The closed-form solution is a mathematical approach to solving linear regression in one step, without iterative optimization. It is derived using concepts from linear algebra and uses the **Normal Equation**:

**θ = (XᵀX)⁻¹ Xᵀ y**

Where:
- `X` is the input feature matrix (augmented with a column of ones for the intercept),
- `y` is the target vector,
- `θ` is the vector of optimal weights (including the bias term).
```python

# FIT MODEL
Input shape        --> (140, 3)  
Add column of 1s   --> (140, 4)  # For bias term  
XᵀX                --> (4, 140) × (140, 4) = (4, 4)  
(XᵀX)⁻¹            --> (4, 4)  
(XᵀX)⁻¹ Xᵀ         --> (4, 4) × (4, 140) = (4, 140)  
(XᵀX)⁻¹ Xᵀ y       --> (4, 140) × (140, 1) = (4, 1)  

# Final parameter vector (bias + weights):  
bias_and_weights   --> (4, 1)  

Extracted parts:  
- bias    = `bias_n_weights[0]`    → scalar  
- weights = `bias_n_weights[1:]`   → (3, 1)  

# MODEL PREDICTION (e.g., X_test shape: (60, 3))  
y_pred = bias + X_test @ weights  
→ bias + (60, 3) × (3, 1) = (60, 1)


This method provides an exact solution for the model parameters and is effective for small to moderately sized datasets. However, for very large datasets, the computation of the matrix inverse can become expensive and numerically unstable.
