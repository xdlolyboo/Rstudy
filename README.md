# Make sure you're inside your Rstudy folder
cd ~/Desktop/Rstudy

# Create .gitignore file
cat <<EOF > .gitignore
.Rhistory
.RData
.Rproj.user/
.DS_Store
EOF

# Create README.md file
cat <<'EOF' > README.md
# 📘 Rstudy – Machine Learning Projects in R

### 🧠 Overview
This repository contains a collection of machine learning projects implemented in **R**.  
Each project explores a different algorithm — from regression to deep learning — with hands-on examples and datasets.

---

## 🗂️ Project Index

| # | Project | Topic | Dataset | Core Libraries |
|---|----------|--------|----------|----------------|
| 1 | [**Linear Regression Study**](#1-linear-regression-study-project) | Regression | Kaggle – Bike Sharing Demand | `tidyverse`, `ggplot2`, `caret` |
| 2 | [**Logistic Regression Study**](#2-logistic-regression-study-project) | Classification | UCI Adult Income | `tidyverse`, `caret`, `ggplot2` |
| 3 | [**K-Nearest Neighbors (KNN) Study**](#3-k-nearest-neighbors-study-project) | Classification | Demo Dataset | `class`, `caret`, `tidyverse` |
| 4 | [**Decision Trees & Random Forests Study**](#4-decision-trees--random-forests-study-project) | Classification | ISLR – College Data | `rpart`, `randomForest`, `ISLR` |
| 5 | [**Support Vector Machines (SVM) Study**](#5-support-vector-machines-study-project) | Classification | Lending Club Loans | `e1071`, `caret`, `ggplot2` |
| 6 | [**K-Means Clustering Study**](#6-k-means-clustering-study-project) | Unsupervised Learning | UCI Wine Quality | `cluster`, `factoextra`, `ggplot2` |
| 7 | [**Neural Networks Study**](#7-neural-networks-study-project) | Deep Learning | UCI Banknote Authentication | `neuralnet`, `caret`, `tidyverse` |

---

## 🧩 Project Summaries

### 1️⃣ Linear Regression Study Project
Predicts **bike rental demand** using linear regression on Kaggle’s dataset.  
Focuses on exploratory analysis and model limitations with nonlinear data.

### 2️⃣ Logistic Regression Study Project
Classifies individuals as earning **≤50K or >50K USD** using the UCI Adult dataset.  
Emphasizes **data cleaning**, encoding, and interpretability.

### 3️⃣ K-Nearest Neighbors (KNN) Study Project
Implements a **KNN classifier** to demonstrate distance-based learning and the effect of `k` on performance.

### 4️⃣ Decision Trees & Random Forests Study Project
Classifies colleges as **Private vs. Public** using tree-based models.  
Shows how Random Forests improve generalization.

### 5️⃣ Support Vector Machines (SVM) Study Project
Predicts **loan repayment** using Lending Club data (2007–2010).  
Compares linear and nonlinear kernels for financial risk modeling.

### 6️⃣ K-Means Clustering Study Project
Clusters **red and white wines** from the UCI dataset.  
Validates clustering quality against actual labels.

### 7️⃣ Neural Networks Study Project
Trains a **Neural Network** to detect forged banknotes.  
Highlights nonlinear modeling strength and superior accuracy.

---

## 🧰 Tech Stack
- **Language:** R  
- **Libraries:** `tidyverse`, `caret`, `ggplot2`, `ISLR`, `rpart`, `randomForest`, `e1071`, `neuralnet`, `cluster`, `factoextra`
- **Tools:** RStudio, Git, GitHub  

