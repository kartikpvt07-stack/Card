#!/usr/bin/env python
# coding: utf-8

# # Cardiovascular Disease Prediction - ML Project
# ## 1. Problem Definition and Dataset Exploration
# **Objective**: Build a machine learning pipeline to predict cardiovascular disease based on clinical and physical features.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Note: The dataset uses ';' as a separator.
df = pd.read_csv('../data/cardio_train.csv', sep=';')
print(df.head())
print(df.info())
print(df.describe())


# ## 2. Data Cleaning and Pre-processing
# - Handle missing values (if any).
# - Identify and handle outliers (e.g., in height, weight, ap_hi, ap_lo).
# - Feature scaling.
# 

# In[2]:


# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Drop the id column as it's not a feature
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Convert age from days to years
df['age'] = (df['age'] / 365).round().astype(int)

# Handle Outliers
# Blood pressure should be within reasonable limits. Let's remove impossible BP values.
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)]
df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 240)]

# Height and weight reasonable bounds
df = df[(df['height'] >= 100) & (df['height'] <= 250)]
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]

print(f"Dataset shape after removing outliers: {df.shape}")

# Separate features and target
X = df.drop('cardio', axis=1)
y = df['cardio']

# Scaling numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X_scaled.head())


# ## 3. Exploratory Data Analysis (EDA)

# In[3]:


# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Distribution of Target Variable
sns.countplot(x='cardio', data=df)
plt.title('Distribution of Target (Cardio)')
plt.show()


# ## 4. Model Creation (Library Implementation)
# We will use Logistic Regression and Random Forest from `sklearn`.

# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression (Library)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)


# ## 5. Implementation Without Library (From Scratch)
# Implementing Logistic Regression using Gradient Descent.

# In[5]:


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Train the scratch model
scratch_model = LogisticRegressionScratch(learning_rate=0.1, epochs=1000)
# Use a subset of data or values to avoid indexing issues
scratch_model.fit(X_train.values, y_train.values)
y_pred_scratch = scratch_model.predict(X_test.values)


# ## 6. Model Evaluation

# In[6]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- Logistic Regression (Library) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(classification_report(y_test, y_pred_log))

print("\n--- Logistic Regression (From Scratch) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_scratch):.4f}")
print(classification_report(y_test, y_pred_scratch))

# Let's also train a Random Forest for comparison
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))


# ## 7. Save Model and Preprocessor for Deployment
# We will save the Random Forest model and the Scaler as they usually perform better.

# In[7]:


import joblib
import os

# Create backend models directory if not exists
os.makedirs('../backend/models', exist_ok=True)

# Save the scaler and the best performing model (let's use the library log_reg for simplicity/speed)
joblib.dump(scaler, '../backend/models/preprocessor.pkl')
joblib.dump(log_reg, '../backend/models/model.pkl')

print("Model and scaler saved to backend/models!")

