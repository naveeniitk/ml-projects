import numpy
import pandas
import logging
from zenml.steps import step


@step(enable_cache=False)
def preprocess_step(dataframe: pandas.DataFrame) -> pandas.DataFrame:

    logging.info(f"Training started for dataframe with shape: {dataframe.shape}")
    logging.info("Training completed successfully.")

    return dataframe
