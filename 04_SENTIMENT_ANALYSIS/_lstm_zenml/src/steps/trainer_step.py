import numpy
import pandas
import logging
from typing import Any
from zenml.steps import step
from models.lstm import LstmClassifier
import config.params as config_params


@step(enable_cache=False)
def trainer_step(
    X_train: numpy.ndarray,
    y_train: pandas.Series,
    hidden_size: int = config_params.HIDDEN_SIZE,
) -> LstmClassifier:

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    y_train = y_train.to_numpy()

    logging.info("Initializing the LSTM model...")
    lstm_model: LstmClassifier = LstmClassifier(
        input_size=len(X_train[0]),
        hidden_size=hidden_size,
        seed=config_params.LSTM_SEED,
    )

    logging.info(f"Starting LSTM training...")
    lstm_model.fit(
        X=X_train,
        y=y_train,
        total_epochs=config_params.LSTM_TOTAL_EPOCHS,
    )

    logging.info("Training completed successfully.")
    return lstm_model
