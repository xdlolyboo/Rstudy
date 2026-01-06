# Neural Networks Study Project
#
# Trains a neural network to detect forged banknotes using the
# UCI Banknote Authentication dataset. Compares neural network
# performance against random forest baseline.
#
# Dataset: bank_note_data.csv (UCI Banknote Authentication)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'Rstudy', 'Neural-Networks-Study-Project')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load Data

df = pd.read_csv(os.path.join(DATA_DIR, 'bank_note_data.csv'))

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nTarget Variable Distribution (Class):")
print(df['Class'].value_counts())
print("  0 = Authentic, 1 = Forged")

# Exploratory Data Analysis

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = ['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy']
colors = {0: '#2ecc71', 1: '#e74c3c'}

for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    for cls in [0, 1]:
        subset = df[df['Class'] == cls][feature]
        label = 'Authentic' if cls == 0 else 'Forged'
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=colors[cls], edgecolor='black')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{feature} Distribution by Class')
    ax.legend()

plt.tight_layout()
plt.savefig('plot1_feature_distributions.png', dpi=100)
plt.show()

# Prepare Data

feature_cols = ['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy']
X = df[feature_cols]
y = df['Class']

np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Model

nn = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='logistic',
    max_iter=1000,
    random_state=101
)
nn.fit(X_train_scaled, y_train)

nn_pred = nn.predict(X_test_scaled)

nn_accuracy = accuracy_score(y_test, nn_pred)
nn_cm = confusion_matrix(y_test, nn_pred)

print("\nNeural Network Results")
print(f"  Architecture: 4 inputs -> 10 hidden (sigmoid) -> 2 outputs")
print("\nConfusion Matrix:")
print(nn_cm)
print(f"\nAccuracy: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")

# Random Forest Comparison

rf = RandomForestClassifier(n_estimators=100, random_state=101, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Results")
print("\nConfusion Matrix:")
print(rf_cm)
print(f"\nAccuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# Model Comparison

print("\nModel Comparison")
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
print(f"Random Forest Accuracy:  {rf_accuracy:.4f}")

if nn_accuracy > rf_accuracy:
    print(f"Neural Network outperforms by {(nn_accuracy - rf_accuracy)*100:.2f}%")
else:
    print(f"Random Forest outperforms by {(rf_accuracy - nn_accuracy)*100:.2f}%")

print("\nClassification Reports:")
print("\n--- Neural Network ---")
print(classification_report(y_test, nn_pred, target_names=['Authentic', 'Forged']))

print("\n--- Random Forest ---")
print(classification_report(y_test, rf_pred, target_names=['Authentic', 'Forged']))

# Confusion matrix comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentic', 'Forged'],
            yticklabels=['Authentic', 'Forged'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Neural Network (Accuracy: {nn_accuracy:.2%})')

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Authentic', 'Forged'],
            yticklabels=['Authentic', 'Forged'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Random Forest (Accuracy: {rf_accuracy:.2%})')

plt.tight_layout()
plt.savefig('plot2_model_comparison.png', dpi=100)
plt.show()

print("\nNeural Networks Study Project Complete!")
