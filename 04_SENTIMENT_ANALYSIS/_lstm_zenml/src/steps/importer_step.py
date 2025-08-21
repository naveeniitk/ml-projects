import numpy
import pandas
import logging
from zenml.steps import step
import config.params as config_params


@step(enable_cache=False)
def importer_step(data_path: str) -> pandas.DataFrame:

    logging.info(f"Importing data from {data_path}")

    # Load the sentiments dataset
    dataframe = pandas.read_csv(data_path, delimiter=",", encoding="utf-8")

    sample_of_datapoints = dataframe[["review", "sentiment"]].sample(
        n=config_params.TOTAL_SAMPLES, random_state=0
    )
    # sample_of_100_datapoints.to_csv('./sentiment-dataset/IMDB/sample_IMDB_dataset.csv', index=False)

    logging.info("Data imported successfully.")
    logging.info(f"Columns found in data: {list(sample_of_datapoints.columns)}")
    logging.info(f"Data: \n{sample_of_datapoints.head(10)}")

    return sample_of_datapoints
