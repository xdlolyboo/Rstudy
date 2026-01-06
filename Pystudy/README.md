# Pystudy - Machine Learning Projects in Python

Python implementations of seven machine learning study projects using Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Index](#project-index)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Descriptions](#project-descriptions)

---

## Overview

This folder contains Python equivalents of the R projects in the Rstudy folder. Each implementation maintains the original logic and educational value while utilizing modern Python data science libraries.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |

---

## Project Index

| Project | Script | Dataset |
|---------|--------|---------|
| Linear Regression | `linear_regression_study.py` | bikeshare.csv |
| Logistic Regression | `logistic_regression_study.py` | adult_sal.csv |
| K-Nearest Neighbors | `knn_study.py` | Iris (sklearn built-in) |
| Decision Trees and Random Forests | `decision_trees_random_forests_study.py` | College.csv |
| Support Vector Machines | `svm_study.py` | loan_data.csv |
| K-Means Clustering | `kmeans_clustering_study.py` | winequality-red/white.csv |
| Neural Networks | `neural_network_study.py` | bank_note_data.csv |

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

Each script can be run from any directory:

```bash
python3 /path/to/DataStudy/Pystudy/Linear-Regression-Study-Project/linear_regression_study.py
```

Or navigate to the project folder first:

```bash
cd ~/DataStudy/Pystudy/K-Nearest-Neighbors-Study-Project
python3 knn_study.py
```

---

## Project Descriptions

### 1. Linear Regression Study Project

Predicts bike rental demand using linear regression. Demonstrates EDA, feature engineering, and model evaluation with R-squared metrics.

### 2. Logistic Regression Study Project

Classifies income levels using UCI Adult dataset. Covers data cleaning, feature encoding, and confusion matrix analysis.

### 3. K-Nearest Neighbors Study Project

Implements KNN classification on Iris dataset with feature standardization and elbow method for optimal K selection.

### 4. Decision Trees and Random Forests Study Project

Classifies colleges as Private/Public. Compares single decision tree with random forest ensemble and analyzes feature importance.

### 5. Support Vector Machines Study Project

Predicts loan repayment using SVM. Includes GridSearchCV for hyperparameter tuning (cost and gamma).

### 6. K-Means Clustering Study Project

Applies unsupervised clustering to wine samples. Compares cluster assignments against actual red/white labels.

### 7. Neural Networks Study Project

Detects forged banknotes using MLPClassifier. Compares neural network accuracy against random forest baseline.
