import torch
import logging
from datetime import datetime
from zenml.steps import step
from models.lstm import LstmClassifier


@step(enable_cache=False)
def save_lstm_model(
    lstm_model: LstmClassifier,
) -> None:
    """
    save lstm model

    Args:
        lstm_model (LstmClassifier):
    """
    timestamp_microseconds = int(datetime.now().timestamp() * 1_000_000)

    logging.info(f"Saving model at: {timestamp_microseconds}")
    torch.save(
        lstm_model,
        f"./saved_models/LSTM_{timestamp_microseconds}.keras",
    )
