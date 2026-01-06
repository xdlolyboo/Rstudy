# Linear Regression Study Project
#
# Predicts bike rental demand using linear regression on the 
# Kaggle Bike Sharing dataset. Focuses on exploratory analysis
# and demonstrates linear regression with feature engineering.
#
# Dataset: bikeshare.csv (Kaggle Bike Sharing Demand)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'Rstudy', 'Linear-Regression-Study-Project')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load Data

bike = pd.read_csv(os.path.join(DATA_DIR, 'bikeshare.csv'))
bike['datetime'] = pd.to_datetime(bike['datetime'])

print("Dataset Shape:", bike.shape)
print("\nFirst 5 rows:")
print(bike.head())

# Exploratory Data Analysis

# Scatter plot: count vs datetime, colored by temperature
fig, ax = plt.subplots(figsize=(14, 6))
scatter = ax.scatter(bike['datetime'], bike['count'], c=bike['temp'], 
                     cmap='YlOrRd', alpha=0.5, s=10)
plt.colorbar(scatter, label='Temperature')
ax.set_xlabel('Date')
ax.set_ylabel('Rental Count')
ax.set_title('Bike Rentals Over Time (colored by Temperature)')
plt.tight_layout()
plt.savefig('plot1_count_vs_datetime.png', dpi=100)
plt.show()

# Correlation between temperature and count
print("\nCorrelation between Temperature and Count:")
print(bike[['temp', 'count']].corr())

# Box plot: count by season
fig, ax = plt.subplots(figsize=(10, 6))
season_colors = {1: '#3498db', 2: '#2ecc71', 3: '#e74c3c', 4: '#9b59b6'}
sns.boxplot(x='season', y='count', data=bike, palette=season_colors.values(), ax=ax)
ax.set_xlabel('Season')
ax.set_ylabel('Rental Count')
ax.set_title('Bike Rentals by Season')
plt.tight_layout()
plt.savefig('plot2_count_by_season.png', dpi=100)
plt.show()

# Feature Engineering

bike['hour'] = bike['datetime'].dt.hour

# Hourly pattern visualization
fig, ax = plt.subplots(figsize=(12, 6))
hourly_avg = bike.groupby('hour')['count'].mean()
ax.bar(hourly_avg.index, hourly_avg.values, color='steelblue', edgecolor='black')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Rental Count')
ax.set_title('Average Bike Rentals by Hour')
plt.tight_layout()
plt.savefig('plot3_hourly_rentals.png', dpi=100)
plt.show()

# Model Training

# Prepare features (exclude redundant columns)
feature_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed', 'hour']
X = bike[feature_cols]
y = bike['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nLinear Regression Results")
print("\nFeature Coefficients:")
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print(coef_df.to_string(index=False))

print(f"\nIntercept: {model.intercept_:.4f}")
print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, predictions, alpha=0.3, s=20)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Count')
ax.set_ylabel('Predicted Count')
ax.set_title(f'Actual vs Predicted (R² = {r2:.4f})')
plt.tight_layout()
plt.savefig('plot4_actual_vs_predicted.png', dpi=100)
plt.show()

print("\nLinear Regression Study Project Complete!")
