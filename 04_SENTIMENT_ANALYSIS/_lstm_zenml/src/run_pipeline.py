import torch
import logging
from config.set_environment import set_ENVIRONMENT
from pipelines.lstm_pipeline import lstm_pipeline

def main():
    logging.info(f"Starting steps in the pipeline")
    lstm_pipeline()
    logging.info(f"Steps completed in the pipeline")


if __name__ == "__main__":

    logging.info("Setting environment variables...")
    set_ENVIRONMENT()

    # =====================================================================
    logging.info("Setting Device to compute...")
    if torch.cuda.is_available():
        logging.info("CUDA available to compute...")
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        logging.info("MPS available to compute...")
        DEVICE = "mps"

    main()
