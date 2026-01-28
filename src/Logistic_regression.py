# ============================================================
# Title: AI-Powered Customer Churn Intelligence System
# Code developer: R. Nanthitha
# Model: Logistic Regression
# ============================================================

# -----------------------------
# Import libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/Telco_customer_churn.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        "Dataset not found. Place 'Telco_customer_churn.xlsx' inside the data/ folder."
    )

print("\nDataset Info:")
print(df.info())
print("\nDataset Preview:")
print(df.head())


# -----------------------------
# Remove irrelevant columns
# -----------------------------
drop_cols = [
    'CustomerID', 'Count', 'Country', 'State', 'City',
    'Zip Code', 'Lat Long', 'Latitude', 'Longitude', 'Churn Reason'
]

df = df.drop(columns=drop_cols, errors='ignore')

print("\nRemaining Columns:")
print(df.columns)
print("Dataset Shape:", df.shape)


# -----------------------------
# Define target and features
# -----------------------------
# Target: 1 = Churned, 0 = Retained
y = df['Churn Value']

# Remove leakage 
X = df.drop(columns=['Churn Value', 'Churn Label', 'Churn Score', 'CLTV'])

print("\nFeatures used for training:")
print(X.columns)


# -----------------------------
# Data Cleaning and Encoding
# -----------------------------
# Strip column spaces
X.columns = X.columns.str.strip()

# Convert Total Charges to numeric
if 'Total Charges' in X.columns:
    X['Total Charges'] = pd.to_numeric(X['Total Charges'], errors='coerce')

# Binary categorical columns
binary_cols = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Phone Service', 'Multiple Lines', 'Online Security',
    'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies', 'Paperless Billing'
]

for col in binary_cols:
    if col in X.columns:
        X[col] = X[col].map({
            'Yes': 1, 'No': 0,
            'Male': 1, 'Female': 0
        }).fillna(0)

# One-Hot Encoding for multi-category features
multi_cols = ['Internet Service', 'Contract', 'Payment Method']
existing_multi_cols = [c for c in multi_cols if c in X.columns]

X = pd.get_dummies(X, columns=existing_multi_cols, drop_first=True)

# Handle missing numeric values
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

print("\nMissing values after preprocessing:", X.isnull().sum().sum())
print("\nFinal feature shape:", X.shape)


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    #80% data for training and 20% data for testing
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
# Model Training
# -----------------------------
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)


# -----------------------------
# Model Evaluation 
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -----------------------------------
# Feature Importance for Business Insight
# -----------------------------------
feature = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
})

feature['Abs_Coefficient'] = feature['Coefficient'].abs()
feature = feature.sort_values(
    by='Abs_Coefficient', ascending=False
).reset_index(drop=True)

print("\nTop Factors Influencing Churn:")
print(feature.head(10))

# -----------------------------
# Baseline Comparison
# -----------------------------

# Initialize dummy classifier (majority class)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_scaled, y_train)
y_dummy_pred = dummy.predict(X_test_scaled)

# Evaluate baseline accuracy
baseline_acc = accuracy_score(y_test, y_dummy_pred)
print("\nBaseline Accuracy (Majority Class):", baseline_acc)

# Compare with Logistic Regression
print("Improvement over baseline:", round(accuracy_score(y_test, y_pred) - baseline_acc, 4))

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
plt.barh(
    feature['Feature'].head(10)[::-1],
    feature['Coefficient'].head(10)[::-1]
)
plt.xlabel("Impact on Churn (+ increases | - reduces)")
plt.title("Top 10 Drivers of Customer Churn")
plt.tight_layout()
plt.show()

# -----------------------------
# ROC-AUC Curve
# -----------------------------
from sklearn.metrics import roc_auc_score, roc_curve

# Get predicted probabilities for positive class
y_probs = model.predict_proba(X_test_scaled)[:,1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_probs)
print("\nROC-AUC Score:", roc_auc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# -----------------------------
# Business Recommendations
# -----------------------------
print("\nBusiness Insights & Retention Strategies:")
print("1) High Monthly Charges --> Offer discounts or loyalty plans")
print("2) Long-term contracts reduce churn --> Promote 1–2 year plans")
print("3) Fiber optic users churn more --> Improve service reliability")
print("4) Tech support reduces churn --> Upsell support packages")
print("5) Payment method impacts churn --> Encourage auto-pay options")

#---------------------------------------
