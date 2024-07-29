import pandas as pd
import logging
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self) -> None:
        pass

    def split_data(self, dataset: pd.DataFrame, target: pd.Series, percentage: float) -> tuple:
        try:
            if not (0 < percentage < 1):
                raise ValueError("Percentage must be between 0 and 1.")
            
            X = dataset
            y = target
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage, random_state=42)
            
            logging.info(f"Data split into train and test sets with test size = {percentage}.")
            return X_train, X_test, y_train, y_test
        except ValueError as ve:
            logging.error(f"Value error: {ve}")
            raise ve
        except Exception as e:
            logging.error("Failed to split data.")
            raise e
