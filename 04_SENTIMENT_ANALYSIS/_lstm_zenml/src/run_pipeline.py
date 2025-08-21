import click
import torch
import logging
from config.set_environment import set_ENVIRONMENT
from pipelines.lstm_pipeline import lstm_pipeline
import config.params as config_params
from config.set_args_from_cli import (
    set_epochs_for_lstm,
    set_total_samples_from_data_for_lstm,
)


def main():
    logging.info(f"Starting steps in the pipeline")
    lstm_pipeline()
    logging.info(f"Steps completed in the pipeline")


@click.command()
@click.option(
    "--epochs", default=config_params.LSTM_TOTAL_EPOCHS, help="#epochs for LSTM"
)
@click.option(
    "--samples", default=config_params.TOTAL_SAMPLES, help="#samples to consider"
)
def set_arguments_from_cli(epochs, samples) -> None:
    logging.info(f"Arguments provided in cli...")
    set_epochs_for_lstm(epochs)
    set_total_samples_from_data_for_lstm(samples)
    main()


if __name__ == "__main__":

    logging.info("Setting environment variables...")
    set_ENVIRONMENT()

    # =====================================================================
    logging.info("Setting Device to compute...")
    if torch.cuda.is_available():
        logging.info("CUDA available to compute...")
        config_params.DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        logging.info("MPS available to compute...")
        config_params.DEVICE = "mps"

    # =====================================================================
    logging.info("Settting Arguments (if any) from (specified in the cli)...")

    set_arguments_from_cli()
