# Evaluation Metrics and Visualization for Avalanche Model

import matplotlib.pyplot as plt
import numpy as np

# Function to calculate evaluation metrics

def evaluate_model(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score

# Function to visualize the results

def visualize_results(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_true)), y_true, color='blue', label='True Values')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.5)
    plt.title('Model Evaluation')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

# Example usage
if __name__ == '__main__':
    y_true = np.array([1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1])
    metrics = evaluate_model(y_true, y_pred)
    print(f'Accuracy: {metrics[0]:.2f}, Precision: {metrics[1]:.2f}, Recall: {metrics[2]:.2f}, F1 Score: {metrics[3]:.2f}')