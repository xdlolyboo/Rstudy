# Support Vector Machines (SVM) Study Project
#
# Predicts loan repayment using SVM on Lending Club data.
# Compares default parameters with tuned RBF kernel using
# grid search for hyperparameter optimization.
#
# Dataset: loan_data.csv (Lending Club)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'Rstudy', 'Support-Vector-Machines-Study-Project')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load Data

loans = pd.read_csv(os.path.join(DATA_DIR, 'loan_data.csv'))

print("Dataset Shape:", loans.shape)
print("\nFirst 5 rows:")
print(loans.head())

# Data Preprocessing

categorical_cols = ['credit.policy', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid']
for col in categorical_cols:
    loans[col] = loans[col].astype('category')

print("\nTarget Variable Distribution (not.fully.paid):")
print(loans['not.fully.paid'].value_counts())

# Exploratory Data Analysis

# FICO score by payment status
fig, ax = plt.subplots(figsize=(12, 6))
fully_paid = loans[loans['not.fully.paid'] == 0]['fico']
not_fully_paid = loans[loans['not.fully.paid'] == 1]['fico']
ax.hist([fully_paid, not_fully_paid], bins=40, 
        label=['Fully Paid', 'Not Fully Paid'],
        color=['green', 'red'], alpha=0.5, edgecolor='black')
ax.set_xlabel('FICO Score')
ax.set_ylabel('Frequency')
ax.set_title('FICO Score Distribution by Payment Status')
ax.legend()
plt.tight_layout()
plt.savefig('plot1_fico_by_payment.png', dpi=100)
plt.show()

# Prepare Data

loans_encoded = pd.get_dummies(loans, columns=['purpose'], drop_first=True)
feature_cols = [col for col in loans_encoded.columns if col != 'not.fully.paid']
X = loans_encoded[feature_cols].astype(float)
y = loans_encoded['not.fully.paid'].astype(int)

np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# SVM with Default Parameters

svm_model = SVC(random_state=101)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

default_accuracy = accuracy_score(y_test, y_pred)
default_cm = confusion_matrix(y_test, y_pred)

print("\nSVM (Default Parameters) Results")
print("\nConfusion Matrix:")
print(default_cm)
print(f"\nAccuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")

# Hyperparameter Tuning

param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.01, 0.1],
    'kernel': ['rbf']
}

print("\nPerforming Grid Search...")
print(f"  C (cost): {param_grid['C']}")
print(f"  gamma: {param_grid['gamma']}")

# Use subset for faster tuning
sample_size = min(2000, len(X_train))
X_train_sample = X_train.iloc[:sample_size]
y_train_sample = y_train.iloc[:sample_size]

grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_sample, y_train_sample)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Tuned SVM Model

best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']

tuned_svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf', random_state=101)
tuned_svm.fit(X_train, y_train)

tuned_pred = tuned_svm.predict(X_test)
tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_cm = confusion_matrix(y_test, tuned_pred)

print(f"\nSVM (Tuned: C={best_C}, gamma={best_gamma}) Results")
print("\nConfusion Matrix:")
print(tuned_cm)
print(f"\nAccuracy: {tuned_accuracy:.4f} ({tuned_accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, tuned_pred, target_names=['Fully Paid', 'Not Fully Paid']))

# Model Comparison

print("\nModel Comparison")
print(f"Default SVM Accuracy: {default_accuracy:.4f}")
print(f"Tuned SVM Accuracy:   {tuned_accuracy:.4f}")
print(f"Improvement: {(tuned_accuracy - default_accuracy)*100:+.2f}%")

print("\nSupport Vector Machines Study Project Complete!")
