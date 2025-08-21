import click
import logging
import config.params as config_params


def set_epochs_for_lstm(epochs) -> None:
    logging.info(f"Setting LSTM_TOTAL_EPOCHS: {epochs}")
    config_params.LSTM_TOTAL_EPOCHS = epochs
    pass


def set_total_samples_from_data_for_lstm(samples) -> None:
    logging.info(f"Setting TOTAL_SAMPLES: {samples}")
    config_params.TOTAL_SAMPLES = samples
    pass
