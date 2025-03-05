import preprocessing
import os

model_path= "/root/code/ml_model/logistic_regression_model.pkl"

print("Preparing features and target...")
df= preprocessing.load_and_preprocess_data()


X = df.drop('Churn', axis=1)
y = df['Churn']

print("\nFeature set shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create and train model
model_lg = LogisticRegression(max_iter=120, random_state=0, n_jobs=20)

# Train model
model_lg.fit(X_train, y_train)

# Make predictions
pred_lg = model_lg.predict(X_test)

# Calculate accuracy
lg = round(accuracy_score(y_test, pred_lg) * 100, 2)

# Print classification report
clf_report = classification_report(y_test, pred_lg)
print(f"Logistic Regression Accuracy: {lg}%\n")
print("Classification Report:\n", clf_report)

# Create and display confusion matrix
plt.figure(figsize=(8, 6))
cm1 = confusion_matrix(y_test, pred_lg)
sns.heatmap(cm1 / np.sum(cm1), annot=True, fmt='.2%', cmap="Reds")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Calculate and print additional metrics
precision = precision_score(y_test, pred_lg)
recall = recall_score(y_test, pred_lg)
f1 = f1_score(y_test, pred_lg)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully.")
    predictions = loaded_model.predict(X_test)
else:
    print(f"Error: Model file '{model_path}' is empty or corrupted!")