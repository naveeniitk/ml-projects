import torch
import numpy
import logging
import pandas
from typing import Dict, Any
from zenml.steps import step
from models.lstm import LstmClassifier
import config.params as config_params


def get_last_model_name() -> str:
    # return "/Users/naveen1.mathur/Desktop/x/sftc/_learning/ml/mlp/ml-projects/04_SENTIMENT_ANALYSIS/_lstm_zenml/src/saved_models/LSTM_1755760417440188.keras"
    # return "./saved_models/LSTM_1755760417440188.keras"
    logging.info(f"Loading : {config_params.LAST_SAVED_MODEL}")
    return str(config_params.LAST_SAVED_MODEL)


@step(enable_cache=False)
def evaluate_model(
    model: LstmClassifier,
    X_test: numpy.ndarray,
    y_true: pandas.Series,
) -> None:
    """
    evaluate_model

    Args:
        model (Any):
        y_true (numpy.ndarray):
        y_pred (numpy.ndarray):

    Returns:
        Dict of metrics

    """
    logging.info(f"Evaluate testing data for final accuracy...")
    # model_name = get_last_model_name()
    # =====================================================================

    # Indicates whether unpickler should be restricted to loading only tensors, primitive
    # types, dictionaries and any types added via torch.serialization.add_safe_globals.
    # logging.info(f"load last saved model...")
    # model: LstmClassifier = torch.load(model_name, weights_only=False)

    logging.info(f"Evaluing model metrics...")

    y_true = y_true.to_numpy()
    logging.info(f"y_true: {y_true}")

    y_pred = model.predict(X=X_test)
    logging.info(f"y_pred: {y_pred}")
