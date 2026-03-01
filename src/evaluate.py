import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Load the trained model
model = joblib.load('path/to/your_model_file')
# Load the test data
# You should replace this with your actual test data loading code
X_test = pd.read_csv('path/to/your_test_data.csv')
# Assuming you have the true labels as well
y_true = X_test['true_labels']
X_test = X_test.drop(columns=['true_labels'])

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
confusion = confusion_matrix(y_true, y_pred)

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(confusion)