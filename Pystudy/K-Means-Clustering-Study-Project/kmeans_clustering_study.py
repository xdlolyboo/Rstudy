# K-Means Clustering Study Project
#
# Applies K-Means clustering to red and white wine samples from
# the UCI Wine Quality dataset. Validates unsupervised clustering
# quality by comparing cluster assignments to actual wine types.
#
# Dataset: winequality-red.csv, winequality-white.csv (UCI Wine Quality)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'Rstudy', 'K-Means-Clustering-Study-Project')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load Data

df_red = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'), sep=';')
df_white = pd.read_csv(os.path.join(DATA_DIR, 'winequality-white.csv'), sep=';')

# Add wine type labels
df_red['label'] = 'red'
df_white['label'] = 'white'

# Combine datasets
wine = pd.concat([df_red, df_white], ignore_index=True)

print("Dataset Shape:", wine.shape)
print("\nWine Type Distribution:")
print(wine['label'].value_counts())

# Exploratory Data Analysis

wine_colors = {'red': '#ae4554', 'white': '#faf7ea'}

# Residual sugar distribution by wine type
fig, ax = plt.subplots(figsize=(12, 6))
for label in ['red', 'white']:
    subset = wine[wine['label'] == label]['residual sugar']
    ax.hist(subset, bins=50, alpha=0.7, label=label.capitalize(),
            color=wine_colors[label], edgecolor='black')
ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Frequency')
ax.set_title('Residual Sugar Distribution by Wine Type')
ax.legend()
plt.tight_layout()
plt.savefig('plot1_residual_sugar.png', dpi=100)
plt.show()

# Citric acid distribution by wine type
fig, ax = plt.subplots(figsize=(12, 6))
for label in ['red', 'white']:
    subset = wine[wine['label'] == label]['citric acid']
    ax.hist(subset, bins=50, alpha=0.7, label=label.capitalize(),
            color=wine_colors[label], edgecolor='black')
ax.set_xlabel('Citric Acid')
ax.set_ylabel('Frequency')
ax.set_title('Citric Acid Distribution by Wine Type')
ax.legend()
plt.tight_layout()
plt.savefig('plot2_citric_acid.png', dpi=100)
plt.show()

# Alcohol distribution by wine type
fig, ax = plt.subplots(figsize=(12, 6))
for label in ['red', 'white']:
    subset = wine[wine['label'] == label]['alcohol']
    ax.hist(subset, bins=50, alpha=0.7, label=label.capitalize(),
            color=wine_colors[label], edgecolor='black')
ax.set_xlabel('Alcohol Content')
ax.set_ylabel('Frequency')
ax.set_title('Alcohol Distribution by Wine Type')
ax.legend()
plt.tight_layout()
plt.savefig('plot3_alcohol.png', dpi=100)
plt.show()

# K-Means Clustering

# Select numeric features (exclude label and quality)
feature_cols = wine.columns.drop(['label', 'quality'])
clus_data = wine[feature_cols]

print(f"\nClustering Features: {list(feature_cols)}")

# Standardize features
scaler = StandardScaler()
clus_data_scaled = scaler.fit_transform(clus_data)

# Apply K-Means with k=2
np.random.seed(101)
kmeans = KMeans(n_clusters=2, random_state=101, n_init=10)
wine['cluster'] = kmeans.fit_predict(clus_data_scaled)

print("\nCluster Centers (standardized):")
centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
print(centers_df.round(3))

print("\nCluster Sizes:")
print(wine['cluster'].value_counts().sort_index())

# Compare Clusters to Actual Labels

wine['label_numeric'] = (wine['label'] == 'white').astype(int)
cm = confusion_matrix(wine['label_numeric'], wine['cluster'])

print("\nConfusion Matrix (Actual vs Cluster):")
print("       Cluster 0  Cluster 1")
print(f"Red:    {cm[0,0]:6d}     {cm[0,1]:6d}")
print(f"White:  {cm[1,0]:6d}     {cm[1,1]:6d}")

accuracy1 = (cm[0,0] + cm[1,1]) / cm.sum()
accuracy2 = (cm[0,1] + cm[1,0]) / cm.sum()
best_accuracy = max(accuracy1, accuracy2)

print(f"\nClustering Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Visualization

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Actual labels
for label in ['red', 'white']:
    subset = wine[wine['label'] == label]
    axes[0].scatter(subset['volatile acidity'], subset['residual sugar'],
                   c=wine_colors[label], label=label.capitalize(), alpha=0.3, s=20)
axes[0].set_xlabel('Volatile Acidity')
axes[0].set_ylabel('Residual Sugar')
axes[0].set_title('Actual Wine Types')
axes[0].legend()

# K-Means clusters
cluster_colors = {0: '#3498db', 1: '#e74c3c'}
for cluster in [0, 1]:
    subset = wine[wine['cluster'] == cluster]
    axes[1].scatter(subset['volatile acidity'], subset['residual sugar'],
                   c=cluster_colors[cluster], label=f'Cluster {cluster}', alpha=0.3, s=20)
axes[1].set_xlabel('Volatile Acidity')
axes[1].set_ylabel('Residual Sugar')
axes[1].set_title('K-Means Clusters')
axes[1].legend()

plt.tight_layout()
plt.savefig('plot4_clusters_comparison.png', dpi=100)
plt.show()

print("\nK-Means Clustering Study Project Complete!")
