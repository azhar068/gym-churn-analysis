# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from scipy.integrate import odeint

# Upload and load data
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]
data = pd.read_csv(filename)

# Display data info
print("Dataset Overview:")
print(data.info())
print("\nFirst few rows:")
print(data.head())

# Data cleaning
data = data.dropna()
print(f"\nShape after dropping missing values: {data.shape}")

# Identify categorical columns and encode them
categorical_cols = ['gender', 'mood_after', 'injury']
for col in categorical_cols:
    if col in data.columns:
        data = pd.get_dummies(data, columns=[col], drop_first=True)

# Define features and target
target_col = 'dropout'
if target_col not in data.columns:
    raise KeyError(f"Target column '{target_col}' not found. Available columns: {data.columns.tolist()}")

features = data.drop(target_col, axis=1)
target = data[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"\nLogistic Regression Accuracy: {lr_accuracy:.3f}")
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))

# Train random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.3f}")
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Feature importance (Random Forest)
feature_importance = pd.DataFrame({
    'feature': features.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance (Random Forest):")
print(feature_importance)

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1]):.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1]):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ODE modeling: membership dynamics
def membership_ode(N, t, A, beta, gamma):
    return A - beta*N - gamma*N

A = 10      # new sign-ups per week
beta = 0.01 # natural expiry rate
gamma = 0.02 # dropout rate
t = np.linspace(0, 52, 52)
N0 = 100
N = odeint(membership_ode, N0, t, args=(A, beta, gamma))

plt.figure(figsize=(10, 6))
plt.plot(t, N, label='Active Members')
plt.xlabel('Week')
plt.ylabel('Active Members')
plt.title('Gym Membership Dynamics')
plt.legend()
plt.show()
