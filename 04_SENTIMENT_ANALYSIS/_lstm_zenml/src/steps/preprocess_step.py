import numpy
import pandas
import logging
from zenml.steps import step
from typing import Tuple
from typing import Annotated
from sklearn import model_selection
from config.params import TEST_SIZE, RANDOM_STATE
from steps.clean_step import clean_step
from steps.embedding_step import embedding_using_vocabulary_building


@step(enable_cache=False)
def preprocess_step(dataframe: pandas.DataFrame) -> Tuple[
    Annotated[pandas.DataFrame, "X_train"],
    Annotated[pandas.DataFrame, "y_train"],
    Annotated[pandas.DataFrame, "X_test"],
    Annotated[pandas.DataFrame, "y_test"],
]:
    logging.info(f"preprocessing started for dataframe with shape: {dataframe.shape}")

    cleaned_data: Tuple[
        Annotated[pandas.DataFrame, "tokenized_features"],
        Annotated[pandas.Series, "tokenized_labels"],
    ] = clean_step(dataframe)

    features: numpy.ndarray = cleaned_data["tokenized_features"]
    labels: numpy.ndarray = cleaned_data["tokenized_labels"]
    
    logging.info("Embedding features using vocabulary building...")
    embedded_features: numpy.ndarray = embedding_using_vocabulary_building(features)
    logging.info(f"Embedding features: {embedded_features[0]}")

    X_train, y_train, X_test, y_test = model_selection.train_test_split(
        embedded_features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    logging.info("Data preprocessing completed successfully.")

    return dataframe
