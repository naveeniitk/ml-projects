import numpy
import pandas
import logging
from zenml.steps import step
from typing import Tuple, List, Dict
from typing import Annotated
from sklearn import model_selection
import config.params as config_params
from steps.clean_step import clean_step

# from steps.embedding_step import embedding_using_vocabulary_building, encode_vocabulary_embedding
from steps.embedding_step import embedding_features_using_sentence_transformer


@step(enable_cache=False)
def preprocess_step(dataframe: pandas.DataFrame) -> Tuple[
    Annotated[numpy.ndarray, "X_train"],
    Annotated[numpy.ndarray, "y_train"],
    Annotated[pandas.Series, "X_test"],
    Annotated[pandas.Series, "y_test"],
]:
    logging.info(f"preprocessing started for dataframe with shape: {dataframe.shape}")
    cleaned_data: Tuple = clean_step(dataframe)

    features: pandas.DataFrame = cleaned_data[0]  # ["tokenized_features"]
    logging.info(f"Features: {features[:5]}")

    embedded_labels: numpy.ndarray = cleaned_data[1]  # ["tokenized_labels"]
    logging.info(f"Labels: {embedded_labels[:5]}")

    # embedding_mapping: Dict[str, int] = embedding_using_vocabulary_building(features)
    # # logging.info(f"Embedding mapping: {embedding_mapping}")

    # embedded_features: numpy.ndarray = encode_vocabulary_embedding(
    #     tokenized_texts=features,
    #     word2idx=embedding_mapping,
    # )

    logging.info( f"word embedding of features using : {config_params.EMBEDDING_MODEL_NAME}" )
    
    embedded_features: numpy.ndarray = embedding_features_using_sentence_transformer(
        features=features,
        max_embedding_size=config_params.SENTENCE_TRANSFORMER_EMBEDDING_SIZE,
    )
    logging.info(f"embedded_features: {embedded_features[:5]}")
    

    X_train, y_train, X_test, y_test = model_selection.train_test_split(
        embedded_features,
        embedded_labels,
        test_size=config_params.TEST_SIZE,
        random_state=config_params.RANDOM_STATE,
        stratify=embedded_labels,
    )

    logging.info("Data preprocessing completed successfully.")

    return X_train, y_train, X_test, y_test
