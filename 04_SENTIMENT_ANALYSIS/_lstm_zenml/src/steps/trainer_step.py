import numpy
import pandas
import logging
from zenml.steps import step


@step(enable_cache=False)
def preprocess_step(dataframe: pandas.DataFrame) -> pandas.DataFrame:

    logging.info(f"preprocessing started for dataframe with shape: {dataframe.shape}")
    logging.info("No preprocessing")
    logging.info("Data preprocessing completed successfully.")

    return dataframe
