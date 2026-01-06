# DataStudy - Machine Learning Projects in R and Python

A comprehensive collection of machine learning study projects implemented in both R and Python. Each project explores a fundamental algorithm with hands-on examples, real-world datasets, and thorough documentation.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Tech Stack](#tech-stack)
4. [Project Index](#project-index)
5. [Installation](#installation)
6. [Usage](#usage)
7. [License](#license)

---

## Overview

This repository contains seven machine learning projects covering supervised learning, unsupervised learning, and neural networks. Each project is implemented in both R and Python, allowing learners to compare approaches across languages.

**Key Features:**
- Parallel implementations in R and Python
- Real-world datasets included with each project
- Well-commented code following language-specific style guides
- Comprehensive exploratory data analysis and visualizations

---

## Repository Structure

```
DataStudy/
├── README.md              # This file
├── Rstudy/                # R implementations
│   ├── README.md
│   ├── Linear-Regression-Study-Project/
│   ├── Logistic-Regression-Study-Project/
│   ├── K-Nearest-Neighbors-Study-Project/
│   ├── Decision-Trees-Random-Forests-Study-Project/
│   ├── Support-Vector-Machines-Study-Project/
│   ├── K-Means-Clustering-Study-Project/
│   └── Neural-Networks-Study-Project/
└── Pystudy/               # Python implementations
    ├── README.md
    ├── Linear-Regression-Study-Project/
    ├── Logistic-Regression-Study-Project/
    ├── K-Nearest-Neighbors-Study-Project/
    ├── Decision-Trees-Random-Forests-Study-Project/
    ├── Support-Vector-Machines-Study-Project/
    ├── K-Means-Clustering-Study-Project/
    ├── Neural-Networks-Study-Project/
```

---

## Tech Stack

### R Implementation (Rstudy)

| Category | Tools |
|----------|-------|
| Language | R (version 4.0+) |
| Data Manipulation | dplyr, tidyr |
| Visualization | ggplot2 |
| Machine Learning | caret, rpart, randomForest, e1071, neuralnet, class |

### Python Implementation (Pystudy)

| Category | Tools |
|----------|-------|
| Language | Python (version 3.8+) |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |

---

## Project Index

| Project | Algorithm Type | Dataset |
|---------|---------------|---------|
| Linear Regression | Regression | Kaggle Bike Sharing Demand |
| Logistic Regression | Binary Classification | UCI Adult Income |
| K-Nearest Neighbors | Classification | Iris Dataset |
| Decision Trees and Random Forests | Ensemble Classification | ISLR College |
| Support Vector Machines | Classification | Lending Club Loans |
| K-Means Clustering | Unsupervised Learning | UCI Wine Quality |
| Neural Networks | Deep Learning | UCI Banknote Authentication |

---

## Installation

### R Setup

1. Install R (version 4.0+) and RStudio
2. Install required packages:

```r
install.packages(c(
  "ggplot2", "dplyr", "caTools", "Amelia", "ISLR",
  "class", "rpart", "rpart.plot", "randomForest",
  "e1071", "cluster", "neuralnet", "MASS"
))
```

### Python Setup

1. Install Python (version 3.8+)
2. Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Clone the Repository

```bash
git clone https://github.com/xdlolyboo/Rstudy.git
cd Rstudy
```

---

## Usage

### Running R Scripts

```r
setwd("~/DataStudy/Rstudy/Linear-Regression-Study-Project")
source("Linear_Regression_Study_Script.R")
```

### Running Python Scripts

```bash
python3 ~/DataStudy/Pystudy/Linear-Regression-Study-Project/linear_regression_study.py
```

Refer to the README files in each subdirectory for detailed documentation:
- [Rstudy README](Rstudy/README.md)
- [Pystudy README](Pystudy/README.md)

---

## License

This project is open source and available under the MIT License.
