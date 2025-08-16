import torch
import pandas
import logging
from config.params import DATA_PATH, DEVICE
from steps.importer_step import importer_step
from steps.preprocess_step import preprocess_step
from steps.trainer_step import trainer_step
from config.set_environment import set_ENVIRONMENT

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

    # =====================================================================
    logging.info("Starting the importer step...")
    try:
        imported_dataframe: pandas.DataFrame = importer_step(data_path=DATA_PATH)
        logging.info("Data imported successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the importer step: {e}")
        raise e
    logging.info("Importer step completed successfully.")

    # =====================================================================
    logging.info("Starting the Preprocessor step...")
    try:
        preprocessed_data = preprocess_step(imported_dataframe)
        logging.info("Data preprocessed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the Preprocessor step: {e}")
        raise e
    logging.info("Preprocessor step completed successfully.")

    # =====================================================================
    logging.info("Training and testing data split successfully.")
    X_train, X_test, y_train, y_test = preprocessed_data

    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    logging.info(f"X_train [0] shape: {X_train[0].shape}")
    logging.info(f"y_train [0] : {y_train.iloc[0]}")
    
    # =====================================================================
    logging.info("Starting the training step...")
    try:
        lstm_model = trainer_step(X_train, y_train)
        logging.info("LSTM Model trained successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the training step: {e}")
        raise e
    logging.info("Training step completed successfully.")
