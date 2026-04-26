import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import joblib
from features import add_features

# Load data
df = pd.read_csv("data/creditcard.csv")
df = add_features(df)
print(df.head())

# Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split (IMPORTANT: stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False
)

# Train
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Default prediction (0.5 threshold)
y_pred_default = (y_proba > 0.5).astype(int)

# Tuned threshold (IMPORTANT)
threshold = 0.3
y_pred_tuned = (y_proba > threshold).astype(int)

# Metrics
print("=== DEFAULT THRESHOLD (0.5) ===")
print(classification_report(y_test, y_pred_default))

print("\n=== TUNED THRESHOLD (0.3) ===")
print(classification_report(y_test, y_pred_tuned))

print("\nROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC:", average_precision_score(y_test, y_proba))

# Save model
joblib.dump(model, "model.pkl")