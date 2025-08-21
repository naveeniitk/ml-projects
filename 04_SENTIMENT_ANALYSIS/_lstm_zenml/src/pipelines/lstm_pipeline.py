import numpy
import pandas
import logging
from datetime import datetime
from typing import Any, Tuple
from zenml import pipeline
import config.params as config_params
from steps.importer_step import importer_step
from steps.preprocess_step import preprocess_step
from steps.trainer_step import trainer_step
from models.lstm import LstmClassifier
from steps.save_models import save_lstm_model


@pipeline(enable_cache=False)
def lstm_pipeline() -> None:
    logging.info(f"Starting the pipeline!!")

    logging.info("Starting the importer step...")
    try:
        imported_dataframe: pandas.DataFrame = importer_step(
            data_path=config_params.DATA_PATH
        )
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

    # =====================================================================
    logging.info("Starting the training step...")
    try:
        lstm_model: LstmClassifier = trainer_step(X_train, y_train)
        logging.info("LSTM Model trained successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the training step: {e}")
        raise e
    logging.info("Training step completed successfully.")

    # =====================================================================
    logging.info(f"Save model info")
    save_lstm_model(lstm_model)
