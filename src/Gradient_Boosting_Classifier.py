# ============================================================
# Title: AI-Powered Customer Churn Intelligence System
# Model: Gradient Boosting Classifier
# Developer: R. Nanthitha
# ============================================================

# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier


# -----------------------------
# Load Dataset 
# -----------------------------
DATA_PATH = "data/Telco_customer_churn.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        "Dataset not found. Place 'Telco_customer_churn.xlsx' inside the data/ folder."
    )

print("\nDataset Loaded Successfully")
print(df.info())


# -----------------------------
# Remove irrelevant columns
# -----------------------------
drop_cols = [
    'CustomerID', 'Count', 'Country', 'State', 'City',
    'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
    'Churn Reason'
]

df = df.drop(columns=drop_cols, errors='ignore')


# -----------------------------
# Define Target and Features
# -----------------------------
y = df['Churn Value']

# Remove leakage columns
X = df.drop(columns=[
    'Churn Value',
    'Churn Label',
    'Churn Score',
    'CLTV'
])


# -----------------------------
# Data Cleaning and Encoding
# -----------------------------
X.columns = X.columns.str.strip()

if 'Total Charges' in X.columns:
    X['Total Charges'] = pd.to_numeric(
        X['Total Charges'],
        errors='coerce'
    )

binary_cols = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Phone Service', 'Multiple Lines',
    'Online Security', 'Online Backup',
    'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies',
    'Paperless Billing'
]

for col in binary_cols:
    if col in X.columns:
        X[col] = X[col].map({
            'Yes': 1, 'No': 0,
            'Male': 1, 'Female': 0
        }).fillna(0)

multi_cols = ['Internet Service', 'Contract', 'Payment Method']
existing_multi_cols = [c for c in multi_cols if c in X.columns]

X = pd.get_dummies(
    X,
    columns=existing_multi_cols,
    drop_first=True
)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

print("\nMissing values after preprocessing:", X.isnull().sum().sum())
print("Final feature shape:", X.shape)


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Baseline Model
# -----------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_scaled, y_train)

baseline_acc = accuracy_score(
    y_test,
    dummy.predict(X_test_scaled)
)

print("\nBaseline Accuracy:", baseline_acc)


# -----------------------------
# Gradient Boosting Model
# -----------------------------
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)


# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = gb_model.predict(X_test_scaled)

print("\nGradient Boosting Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print(
    "Improvement over baseline:",
    round(accuracy_score(y_test, y_pred) - baseline_acc, 4)
)


# -----------------------------
# ROC-AUC Curve
# -----------------------------
y_probs = gb_model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_probs)
print("\nROC-AUC Score:", roc_auc)

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Gradient Boosting (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Gradient Boosting")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Feature Importance
# -----------------------------
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Factors Influencing Churn (Gradient Boosting):")
print(feature_importance.head(10))


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance['Feature'].head(10)[::-1],
    feature_importance['Importance'].head(10)[::-1]
)
plt.xlabel("Relative Importance")
plt.title("Top 10 Drivers of Customer Churn - Gradient Boosting")
plt.tight_layout()
plt.show()


# -----------------------------
# Business Recommendations
# -----------------------------
print("\nBusiness Insights & Retention Strategies:")
print("1) Long-tenure customers are loyal, so introduce loyalty rewards.")
print("2) Fiber optic users show higher churn; improving service quality will reduce this churn.")
print("3) Long-term contracts reduce churn; therefore, promoting 1–2 year plans is recommended.")
print("4) Personalized discounts will reduce churn caused by high monthly charges.")
print("5) Support services improve retention, so bundling tech support is recommended.")
