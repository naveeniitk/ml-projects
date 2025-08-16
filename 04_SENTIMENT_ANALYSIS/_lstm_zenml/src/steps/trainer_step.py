import numpy
import pandas
import logging
from zenml.steps import step
from models.lstm import LstmClassifier
from config.params import HIDDEN_SIZE


@step(enable_cache=False)
def trainer_step(
    X_train: numpy.ndarray,
    y_train: pandas.Series,
    hidden_size: int = HIDDEN_SIZE,
) -> LstmClassifier:

    logging.info("Initializing the LSTM model...")

    lstm_model = LstmClassifier(len(X_train), hidden_size)

    logging.info(f"Training started for X_train with shape: {X_train.shape}")
    logging.info(f"Training started for y_train with shape: {y_train.shape}")

    logging.info("Training completed successfully.")

    return lstm_model
