import numpy
import pandas
import logging
from zenml.steps import step
from models.lstm import LstmClassifier


@step(enable_cache=False)
def trainer_step(
    X_train: numpy.ndarray,
    y_train: pandas.Series,
) -> None:

    logging.info("Initializing the LSTM model...")
    lstm_model = LstmClassifier()
    
    logging.info(f"Training started for X_train with shape: {X_train.shape}")
    logging.info(f"Training started for y_train with shape: {y_train.shape}")
        
    
    logging.info("Training completed successfully.")

    return None
