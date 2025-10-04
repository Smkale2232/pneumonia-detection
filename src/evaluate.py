# src/evaluate.py - FIXED VERSION
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
import json
import os


class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def evaluate_model(self, test_dataset):
        """Step 5: Comprehensive evaluation - FIXED VERSION"""
        print("Evaluating model on test set...")

        # Get predictions
        y_pred = self.model.predict(test_dataset)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Get true labels
        y_true = np.concatenate([y for x, y in test_dataset], axis=0)

        # Calculate basic metrics using model.evaluate
        # Use try-except since metrics might vary
        try:
            # Try to get all metrics
            evaluation_results = self.model.evaluate(test_dataset, verbose=0)
            test_loss = evaluation_results[0]
            test_accuracy = evaluation_results[1]

            # If we have more metrics, use them, otherwise calculate manually
            if len(evaluation_results) >= 5:
                test_precision = evaluation_results[2]
                test_recall = evaluation_results[3]
                test_auc = evaluation_results[4]
            else:
                # Calculate manually using sklearn
                test_precision = precision_score(y_true, y_pred_binary)
                test_recall = recall_score(y_true, y_pred_binary)
                test_auc = roc_auc_score(y_true, y_pred)

        except Exception as e:
            print(f"Using manual metric calculation: {e}")
            test_loss, test_accuracy = self.model.evaluate(
                test_dataset, verbose=0)
            test_precision = precision_score(y_true, y_pred_binary)
            test_recall = recall_score(y_true, y_pred_binary)
            test_auc = roc_auc_score(y_true, y_pred)

        # Print results
        print(f"\n=== Test Set Evaluation ===")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"AUC: {test_auc:.4f}")

        # Classification report
        print(f"\n=== Classification Report ===")
        print(classification_report(y_true, y_pred_binary,
                                    target_names=['Normal', 'Pneumonia']))

        # Save metrics
        self.save_metrics(test_loss, test_accuracy, test_precision,
                          test_recall, test_auc, y_true, y_pred_binary)

        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred_binary)

        return y_true, y_pred, y_pred_binary

    def plot_confusion_matrix(self, y_true, y_pred_binary):
        """Step 5: Confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save plot
        os.makedirs('results/training_plots', exist_ok=True)
        plt.savefig('results/training_plots/confusion_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def save_metrics(self, loss, accuracy, precision, recall, auc, y_true, y_pred_binary):
        """Step 5: Save evaluation metrics"""
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc),
            'f1_score': float(f1)
        }

        os.makedirs('results/metrics', exist_ok=True)
        with open('results/metrics/test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save classification report
        report = classification_report(y_true, y_pred_binary,
                                       target_names=['Normal', 'Pneumonia'])

        with open('results/metrics/classification_report.txt', 'w') as f:
            f.write(report)

        print(f"✓ Metrics saved to results/metrics/")
