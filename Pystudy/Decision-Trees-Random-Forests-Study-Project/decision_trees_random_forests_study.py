# Decision Trees and Random Forests Study Project
#
# Classifies colleges as Private or Public using decision tree
# and random forest models. Demonstrates how ensemble methods
# improve generalization over single trees.
#
# Dataset: College.csv (ISLR equivalent)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load Data

df = pd.read_csv(os.path.join(SCRIPT_DIR, 'College.csv'))

if 'Unnamed: 0' in df.columns:
    df = df.set_index('Unnamed: 0')
elif df.columns[0] not in ['Private', 'Apps', 'Accept']:
    df = df.set_index(df.columns[0])

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nPrivate/Public Distribution:")
print(df['Private'].value_counts())

# Exploratory Data Analysis

# Room & Board vs Graduation Rate
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'Yes': '#3498db', 'No': '#e74c3c'}
for private_status in ['Yes', 'No']:
    subset = df[df['Private'] == private_status]
    ax.scatter(subset['Room.Board'], subset['Grad.Rate'], 
               c=colors[private_status], label=f'Private={private_status}', 
               alpha=0.5, s=50)
ax.set_xlabel('Room & Board Cost')
ax.set_ylabel('Graduation Rate')
ax.set_title('Room & Board vs Graduation Rate by Private Status')
ax.legend()
plt.tight_layout()
plt.savefig('plot1_room_board_vs_grad_rate.png', dpi=100)
plt.show()

# Full-time undergrad distribution
fig, ax = plt.subplots(figsize=(10, 6))
private_yes = df[df['Private'] == 'Yes']['F.Undergrad']
private_no = df[df['Private'] == 'No']['F.Undergrad']
ax.hist([private_yes, private_no], bins=50, label=['Private=Yes', 'Private=No'],
        color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Full-time Undergraduates')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Full-time Undergraduates by Private Status')
ax.legend()
plt.tight_layout()
plt.savefig('plot2_f_undergrad_distribution.png', dpi=100)
plt.show()

# Fix: Cap Grad.Rate at 100
df.loc[df['Grad.Rate'] > 100, 'Grad.Rate'] = 100

# Prepare Data

X = df.drop('Private', axis=1)
y = df['Private']

np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Decision Tree Model

tree = DecisionTreeClassifier(random_state=101, max_depth=5)
tree.fit(X_train, y_train)

tree_preds = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_preds)
tree_cm = confusion_matrix(y_test, tree_preds)

print("\nDecision Tree Results")
print("\nConfusion Matrix:")
print(tree_cm)
print(f"\nAccuracy: {tree_accuracy:.4f} ({tree_accuracy*100:.2f}%)")

# Random Forest Model

rf_model = RandomForestClassifier(n_estimators=100, random_state=101, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_cm = confusion_matrix(y_test, rf_preds)

print("\nRandom Forest Results")
print("\nConfusion Matrix:")
print(rf_cm)
print(f"\nAccuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, rf_preds, target_names=['Public (No)', 'Private (Yes)']))

# Feature Importance

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(10)
ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Top 10 Feature Importances (Random Forest)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('plot3_feature_importance.png', dpi=100)
plt.show()

# Model Comparison

print("\nModel Comparison")
print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Improvement: {(rf_accuracy - tree_accuracy)*100:.2f}%")

print("\nDecision Trees & Random Forests Study Project Complete!")
