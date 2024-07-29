import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import logging
import joblib

class ModelTrainer:
    def __init__(self, training_set: pd.DataFrame, 
                 target_training_set: pd.Series, model: LogisticRegression):
        self.training_set = training_set
        self.target_training_set = target_training_set
        self.model = model  # Use instance variable for the model

    def train_model(self):
        if self.training_set is None or self.target_training_set is None:
            raise ValueError("Data has not been provided. Ensure training_set and target_training_set are set.")
        
        logging.info("Training the model")
        self.model.fit(self.training_set, self.target_training_set)
        logging.info("Model Trained Successfuly, passing the results to evaluate")

    
    def get_trained_model(self):
        return self.model
    
    def save_trained_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save. Ensure the model is trained before saving.")
        
        try:
            joblib.dump(self.model, filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error in saving the model: {e}")
            raise
