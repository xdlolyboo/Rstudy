# Rstudy - Machine Learning Projects in R

A collection of machine learning study projects implemented in R. Each project explores a different algorithm with hands-on examples, real-world datasets, and comprehensive documentation.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Index](#project-index)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Descriptions](#project-descriptions)
7. [License](#license)

---

## Overview

This repository contains seven machine learning projects covering fundamental algorithms in supervised learning, unsupervised learning, and neural networks. Each project includes:

- Well-commented R scripts following tidyverse style guidelines
- Real-world datasets for practical learning
- Exploratory data analysis with ggplot2 visualizations
- Model training, evaluation, and interpretation

The projects are designed for educational purposes, suitable for students and practitioners learning machine learning concepts in R.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | R (version 4.0+) |
| Data Manipulation | dplyr, tidyr |
| Visualization | ggplot2 |
| Machine Learning | caret, rpart, randomForest, e1071, neuralnet, class |
| Utilities | caTools, ISLR, MASS, Amelia, cluster |

---

## Project Index

| Project | Algorithm | Dataset | Key Libraries |
|---------|-----------|---------|---------------|
| Linear Regression | Linear Regression | Kaggle Bike Sharing | ggplot2, dplyr |
| Logistic Regression | Logistic Regression | UCI Adult Income | caTools, Amelia |
| K-Nearest Neighbors | KNN Classification | Iris (built-in) | class, ISLR |
| Decision Trees and Random Forests | Tree-based Classification | ISLR College | rpart, randomForest |
| Support Vector Machines | SVM Classification | Lending Club Loans | e1071 |
| K-Means Clustering | Unsupervised Clustering | UCI Wine Quality | cluster |
| Neural Networks | Feedforward Neural Network | UCI Banknote Auth | neuralnet |

---

## Installation

### Prerequisites

Ensure R (version 4.0 or higher) and RStudio are installed on your system.

### Install Required Packages

Run the following command in R to install all required packages:

```r
install.packages(c(
  "ggplot2",
  "dplyr",
  "caTools",
  "Amelia",
  "corrplot",
  "ISLR",
  "class",
  "rpart",
  "rpart.plot",
  "randomForest",
  "e1071",
  "cluster",
  "factoextra",
  "neuralnet",
  "MASS"
))
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/Rstudy.git
cd Rstudy
```

---

## Usage

Each project is contained in its own directory with an R script and the necessary dataset(s).

1. Open RStudio
2. Set the working directory to the desired project folder
3. Open the R script and run it section by section

Example:

```r
setwd("~/Rstudy/Linear-Regression-Study-Project")
source("Linear_Regression_Study_Script.R")
```

---

## Project Descriptions

### 1. Linear Regression Study Project

**Directory:** `Linear-Regression-Study-Project/`

Predicts bike rental demand using linear regression on the Kaggle Bike Sharing dataset. The project demonstrates exploratory data analysis, feature engineering (extracting temporal features), and the limitations of linear models with nonlinear data.

**Key Concepts:**
- Correlation analysis
- Feature engineering from datetime
- Linear regression diagnostics

---

### 2. Logistic Regression Study Project

**Directory:** `Logistic-Regression-Study-Project/`

Classifies individuals as earning above or below 50K USD annually using the UCI Adult Income dataset. Emphasizes data cleaning, categorical variable encoding, and model evaluation metrics.

**Key Concepts:**
- Handling missing values
- Feature grouping and transformation
- Accuracy, precision, and recall

---

### 3. K-Nearest Neighbors Study Project

**Directory:** `K-Nearest-Neighbors-Study-Project/`

Implements KNN classification on the Iris dataset to demonstrate distance-based learning. Includes feature standardization and the elbow method for selecting optimal K.

**Key Concepts:**
- Feature scaling
- Distance-based classification
- Hyperparameter selection (K)

---

### 4. Decision Trees and Random Forests Study Project

**Directory:** `Decision-Trees-Random-Forests-Study-Project/`

Classifies colleges as Private or Public using decision trees and random forests on the ISLR College dataset. Demonstrates how ensemble methods improve generalization.

**Key Concepts:**
- Decision tree visualization
- Ensemble learning
- Feature importance analysis

---

### 5. Support Vector Machines Study Project

**Directory:** `Support-Vector-Machines-Study-Project/`

Predicts loan repayment using SVM on Lending Club data. Compares default parameters with tuned RBF kernel using grid search optimization.

**Key Concepts:**
- Kernel methods
- Hyperparameter tuning (cost, gamma)
- Financial risk prediction

---

### 6. K-Means Clustering Study Project

**Directory:** `K-Means-Clustering-Study-Project/`

Applies unsupervised K-Means clustering to red and white wine samples from the UCI Wine Quality dataset. Validates clustering quality against actual labels.

**Key Concepts:**
- Unsupervised learning
- Cluster evaluation
- Feature distribution analysis

---

### 7. Neural Networks Study Project

**Directory:** `Neural-Networks-Study-Project/`

Trains a neural network to detect forged banknotes using the UCI Banknote Authentication dataset. Compares neural network performance against a random forest baseline.

**Key Concepts:**
- Feedforward neural networks
- Nonlinear pattern recognition
- Model comparison

---

## License

This project is open source and available under the MIT License.
