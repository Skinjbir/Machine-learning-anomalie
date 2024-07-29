import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

class Evaluator:
    def __init__(self, test_set: pd.DataFrame, target: pd.Series):
        self.test_set = test_set
        self.target = target

    def evaluate(self, model):
        if self.test_set is None or self.target is None:
            raise ValueError("Test data and target must be provided.")
        
        if model is None:
            raise ValueError("Model must be provided for evaluation.")
        
        # Predict using the provided model
        predictions = model.predict(self.test_set)

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.target, predictions)
        precision = precision_score(self.target, predictions)
        recall = recall_score(self.target, predictions)
        f1 = f1_score(self.target, predictions)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
