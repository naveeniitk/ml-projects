import numpy
import pandas
import logging
from typing import Any, Tuple
from zenml import pipeline
from config.params import DATA_PATH
from steps.importer_step import importer_step
from steps.preprocess_step import preprocess_step
from steps.trainer_step import trainer_step


@pipeline(enable_cache=False)
def lstm_pipeline() -> None:
    logging.info(f"Starting the pipeline!!")

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
        preprocessed_data: Tuple = preprocess_step(imported_dataframe)
        logging.info("Data preprocessed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the Preprocessor step: {e}")
        raise e
    logging.info("Preprocessor step completed successfully.")

    print(f"type(preprocessed_data): {type(preprocessed_data)}")

    # =====================================================================
    logging.info("Training and testing data split successfully.")
    X_train, X_test, y_train, y_test = preprocessed_data

    # logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # logging.info(f"X_train [0] shape: {X_train[0].shape}")
    # logging.info(f"y_train [0] : {y_train.iloc[0]}")

    # =====================================================================
    logging.info("Starting the training step...")
    try:
        lstm_model: Any = trainer_step(X_train, y_train)
        logging.info("LSTM Model trained successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the training step: {e}")
        raise e
    logging.info("Training step completed successfully.")
