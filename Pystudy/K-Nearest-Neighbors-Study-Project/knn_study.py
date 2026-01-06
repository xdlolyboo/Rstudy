# K-Nearest Neighbors (KNN) Study Project
#
# Implements KNN classification on the Iris dataset to demonstrate
# distance-based learning. Includes standardization, elbow method
# for optimal K selection, and error rate visualization.
#
# Dataset: Iris (sklearn built-in)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load Data

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='Species')

print("Dataset Shape:", X.shape)
print("\nFeature Statistics:")
print(X.describe())
print("\nSpecies Distribution:")
species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
y_named = y.map(species_names)
print(y_named.value_counts())

# Data Preparation

# Standardize features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Feature Statistics (should have mean~0, std~1):")
print(pd.DataFrame(X_scaled, columns=iris.feature_names).describe().loc[['mean', 'std']])

# Train-Test Split

np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=101
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Initial KNN with K=1

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print("\nKNN (K=1) Results")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")

# Elbow Method: Find Optimal K

error_rates = []
k_values = range(1, 11)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate = np.mean(pred != y_test)
    error_rates.append(error_rate)
    print(f"K={k:2d} | Error Rate: {error_rate:.4f} | Accuracy: {1-error_rate:.4f}")

error_df = pd.DataFrame({'K': k_values, 'Error_Rate': error_rates})

# Plot error rate vs K
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(error_df['K'], error_df['Error_Rate'], 'ro-', markersize=10, linewidth=2)
ax.set_xlabel('K Value')
ax.set_ylabel('Error Rate')
ax.set_title('KNN Error Rate vs K (Elbow Method)')
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig('plot1_error_rate_vs_k.png', dpi=100)
plt.show()

# Final Model with Optimal K

optimal_k = error_df.loc[error_df['Error_Rate'].idxmin(), 'K']
min_error = error_df['Error_Rate'].min()

print(f"\nOptimal K: {optimal_k}")
print(f"Minimum Error Rate: {min_error:.4f}")
print(f"Maximum Accuracy: {1-min_error:.4f}")

# Train final model
knn_final = KNeighborsClassifier(n_neighbors=int(optimal_k))
knn_final.fit(X_train, y_train)
final_pred = knn_final.predict(X_test)

print(f"\nKNN (K={int(optimal_k)}) Final Results")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))
print("\nClassification Report:")
print(classification_report(y_test, final_pred, target_names=iris.target_names))

print("\nK-Nearest Neighbors Study Project Complete!")
