import numpy
import pandas
import logging
from zenml.steps import step
from models.lstm import LstmClassifier
from config.params import HIDDEN_SIZE, LSTM_SEED
from typing import Any


@step(enable_cache=False)
def trainer_step(
    X_train: numpy.ndarray,
    y_train: pandas.Series,
    hidden_size: int = HIDDEN_SIZE,
) -> LstmClassifier:

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    logging.info("Initializing the LSTM model...")
    lstm_model: Any = LstmClassifier(len(X_train), hidden_size, seed=LSTM_SEED)

    logging.info(f"Training started for X_train with shape: {X_train.shape}")
    logging.info(f"Training started for y_train with shape: {y_train.shape}")

    logging.info("Training completed successfully.")

    return lstm_model
