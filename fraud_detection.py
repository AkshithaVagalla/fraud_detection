import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load Data (Fixing file path)
df = pd.read_csv(r"C:\Users\Akshitha\OneDrive\small projects -AI\fradud\creditcard.csv")

# Check class distribution
print(df["Class"].value_counts())

# Split Features & Labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle Imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# Ensure "models/" directory exists before saving
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")

