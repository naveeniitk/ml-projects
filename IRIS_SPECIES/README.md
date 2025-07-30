# Iris Species Classification and Visualization

This project explores the classic Iris dataset using a variety of data visualizations and machine learning models to classify iris flower species based on sepal and petal measurements.

## Dataset

The dataset used is [`Iris.csv`](./data/Iris.csv), which contains 150 samples across three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Each sample includes the following features:

- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

## Project Structure

- **Data Loading and Preprocessing**: Read and clean the Iris dataset using pandas.
- **Exploratory Data Analysis (EDA)**:
  - Univariate, bivariate, and multivariate visualizations using pandas, matplotlib, and seaborn.
  - Distributions and class balance checks.
- **Feature Visualization**:

  - Scatter plots, box plots, strip plots, violin plots, KDE plots
  - Pair plots, Andrews curves, parallel coordinates, and radviz plots

- **Model Training and Evaluation**:
  - Train/test split and 7-fold cross-validation
  - Classifiers used:
    - K-Nearest Neighbors (KNN)
    - Naive Bayes (GaussianNB)
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
  - Evaluation metric: average accuracy across folds
