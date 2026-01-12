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


## Telco Customer Churn Prediction
This project predicts customer churn for a telecommunications company using supervised machine learning models. The objective is to identify customers likely to stop using the service based on their demographics and usage behavior.

Used SMOTE as the Telco Churn dataset contains a significant class imbalance: the number of customers who do **not** churn ("No") is much higher than those who do ("Yes"). This imbalance can lead to models that perform well on accuracy but fail to correctly predict churners, resulting in poor recall.

## Sentiment Analysis
This project implements a sentiment analysis pipeline utilizing Long Short-Term Memory (LSTM) neural networks to classify textual sentiment. The implementation leverages multiple benchmark datasets including IMDB movie reviews, Twitter sentiment corpus, and Amazon product reviews, enabling robust evaluation across different text styles and domains.

The architecture employs recurrent neural networks with LSTM cells to capture long-range dependencies and contextual nuances in natural language. The project incorporates ZenML for experiment tracking, model versioning, and pipeline orchestration, facilitating reproducible machine learning workflows.

## OCR Document Parser
This project presents an end-to-end document intelligence system that combines optical character recognition (OCR) with advanced document understanding models to extract structured information from unstructured documents. The system employs a multi-stage pipeline utilizing **EasyOCR** and **Tesseract** for initial text extraction, followed by fine-tuned **LayoutLMv3** models for document layout understanding and named entity recognition.

The architecture leverages vision-language models that simultaneously process both textual content and spatial layout information. The system demonstrates practical application in invoice and bill processing, extracting key entities such as customer information, tax identifiers, transaction dates, product details, and monetary values, converting them into machine-readable JSON representations.

## Agentic AI
This project implements an autonomous AI agent framework built on Google's Gemini API, demonstrating agentic AI systems that can independently plan, execute, and adapt their behavior through tool use. The agent operates through a function calling paradigm, autonomously selecting and invoking appropriate tools from a predefined set of capabilities including file system operations, code execution, and information retrieval.

The architecture enables the agent to engage in multi-step reasoning and iterative problem-solving, making autonomous decisions about which operations to perform based on high-level user instructions. The implementation includes comprehensive security measures such as path validation, directory traversal prevention, execution timeouts, and controlled working environments to ensure safe operation.

---