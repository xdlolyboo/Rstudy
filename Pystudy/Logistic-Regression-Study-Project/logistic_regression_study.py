# Logistic Regression Study Project
#
# Classifies individuals as earning above or below 50K USD using
# logistic regression on the UCI Adult Income dataset. Emphasizes
# data cleaning, categorical encoding, and model evaluation.
#
# Dataset: adult_sal.csv (UCI Adult Income)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'Rstudy', 'Logistic-Regression-Study-Project')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load Data

adult = pd.read_csv(os.path.join(DATA_DIR, 'adult_sal.csv'))
if 'Unnamed: 0' in adult.columns:
    adult = adult.drop('Unnamed: 0', axis=1)

print("Dataset Shape:", adult.shape)
print("\nFirst 5 rows:")
print(adult.head())

# Data Cleaning: Employment Type

def combine_employer(job):
    if job in ['Never-worked', 'Without-pay']:
        return 'Unemployed'
    elif job in ['Local-gov', 'State-gov']:
        return 'SL-gov'
    elif job in ['Self-emp-inc', 'Self-emp-not-inc']:
        return 'self-emp'
    else:
        return job

adult['type_employer'] = adult['type_employer'].apply(combine_employer)

# Data Cleaning: Marital Status

def group_marital(status):
    if status in ['Separated', 'Divorced', 'Widowed']:
        return 'Not-Married'
    elif status == 'Never-married':
        return status
    else:
        return 'Married'

adult['marital'] = adult['marital'].apply(group_marital)

# Data Cleaning: Country to Region

Asia = ['China', 'Hong', 'India', 'Iran', 'Cambodia', 'Japan', 'Laos',
        'Philippines', 'Vietnam', 'Taiwan', 'Thailand']
North_America = ['Canada', 'United-States', 'Puerto-Rico']
Europe = ['England', 'France', 'Germany', 'Greece', 'Holand-Netherlands',
          'Hungary', 'Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland', 'Yugoslavia']
Latin_South_America = ['Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',
                       'El-Salvador', 'Guatemala', 'Haiti', 'Honduras',
                       'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
                       'Peru', 'Jamaica', 'Trinadad&Tobago']

def group_country(country):
    if country in Asia:
        return 'Asia'
    elif country in North_America:
        return 'North_America'
    elif country in Europe:
        return 'Europe'
    elif country in Latin_South_America:
        return 'Latin_South_America'
    else:
        return 'Other'

adult['region'] = adult['country'].apply(group_country)
adult = adult.drop('country', axis=1)

# Handle Missing Values

adult = adult.replace(' ?', np.nan).replace('?', np.nan)
print("\nMissing Values Before Cleaning:")
print(adult.isnull().sum()[adult.isnull().sum() > 0])

adult = adult.dropna()
print(f"\nDataset shape after dropping missing values: {adult.shape}")

# Exploratory Data Analysis

fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=adult, x='age', hue='income', bins=40, ax=ax)
ax.set_title('Age Distribution by Income Level')
plt.tight_layout()
plt.savefig('plot1_age_distribution.png', dpi=100)
plt.show()

# Model Training

adult['income'] = (adult['income'].str.strip() == '>50K').astype(int)

# One-hot encode categorical variables
categorical_cols = adult.select_dtypes(include=['object']).columns
adult_encoded = pd.get_dummies(adult, columns=categorical_cols, drop_first=True)

X = adult_encoded.drop('income', axis=1)
y = adult_encoded['income']

print(f"\nFeature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model Evaluation

predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print("\nLogistic Regression Results")
print("\nConfusion Matrix:")
print(cm)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))

# Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
plt.tight_layout()
plt.savefig('plot2_confusion_matrix.png', dpi=100)
plt.show()

print("\nLogistic Regression Study Project Complete!")
