import logging
import pandas
from config.params import DATA_PATH
from steps.importer_step import importer_step
from steps.preprocess_step import preprocess_step
from steps.trainer_step import trainer_step

if __name__ == "__main__":

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
    X_train, X_test, y_train, y_test = preprocessed_data

    # =====================================================================
    logging.info("Starting the training step...")
    try:
        trained_lstm_model = trainer_step(X_train, y_train)
        logging.info("LSTM Model trained successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the training step: {e}")
        raise e
    logging.info("Training step completed successfully.")
