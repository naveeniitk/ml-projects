import numpy
import pandas
import logging
import re as pythonRegex
from zenml.steps import step
from autocorrect import Speller
from typing import Annotated, Tuple

# The Natural Language Toolkit (NLTK) is an open
# source Python library for Natural Language Processing
import nltk
from nltk import tokenize  # word_tokenize

# Download required NLTK data
# nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("wordnet")

spellings = Speller(lang="en")
lemmatizer = nltk.WordNetLemmatizer()


def clean_string(text: str) -> str:
    new_text = text.strip()  # to remove leading and trailing spaces
    rm_numbers_from_text = pythonRegex.sub(r"[0-9]", "", new_text).strip()
    text_to_lowercase = rm_numbers_from_text.lower().strip()
    # Removing hashtags and mentions
    rm_mentions_from_text = pythonRegex.sub(
        "@[A-Za-z0-9_]+", "", text_to_lowercase
    ).strip()
    rm_hastags_from_text = pythonRegex.sub(
        "#[A-Za-z0-9_]+", "", rm_mentions_from_text
    ).strip()
    # Removing links
    rm_links_from_text1 = pythonRegex.sub(r"http\S+", "", rm_hastags_from_text).strip()
    # \S = non-whitespace character
    rm_links_from_text = pythonRegex.sub(r"www.\S+", "", rm_links_from_text1).strip()
    # removing tags like <br> and <p>
    rm_tags_from_text = pythonRegex.sub(r"<.*?>", "", rm_links_from_text).strip()

    # Removing '
    final_text = pythonRegex.sub("'", "", rm_tags_from_text).strip()
    rm_spaces_from_text = pythonRegex.sub(r"\s+", " ", final_text).strip()
    # return rm_spaces_from_text;

    text_tokenization = tokenize.word_tokenize(rm_spaces_from_text)
    text_correction = [spellings(token) for token in text_tokenization]

    # lemmatization means to reduce a word to its base or root form
    # e.g., "running" becomes "run", "better" becomes "good"
    text_lemmatization = [lemmatizer.lemmatize(token) for token in text_correction]
    return text_lemmatization


def tokenize_string(text: str) -> numpy.ndarray:
    splitted_text = text.split(" ")
    final_data = [token for token in splitted_text if len(token) > 1]
    return final_data


def clean_features(data: pandas.DataFrame) -> pandas.DataFrame:
    data = data.apply(clean_string)
    return pandas.DataFrame.copy(data)


def clean_labels(data: pandas.Series) -> pandas.Series:
    return pandas.Series.copy(data)


def tokenize_labels(labels: pandas.Series) -> pandas.Series:
    new_labels: pandas.Series = labels.y.map({"positive": 1, "negative": 0})
    return pandas.Series.copy(new_labels)


@step(enable_cache=False)
def clean_step(
    dataframe: pandas.DataFrame,
) -> Tuple[
    Annotated[pandas.DataFrame, "cleaned_features"],
    Annotated[pandas.Series, "cleaned_labels"],
]:
    logging.info(f"Cleaning started for dataframe with shape: {dataframe.shape}")
    data: pandas.DataFrame = dataframe["review"]
    labels: pandas.Series = dataframe["sentiment"]

    logging.info("Cleaning features...")
    cleaned_data: pandas.DataFrame = clean_features(data)

    logging.info("Tokenizing features...")
    tokenized_features: pandas.DataFrame = cleaned_data.apply(tokenize_string)

    logging.info("Cleaning labels...")
    cleaned_labels: pandas.Series = clean_labels(labels)

    logging.info("Tokenizing labels...")
    tokenized_labels: pandas.Series = tokenize_labels(cleaned_labels)

    logging.info("Cleaning/Tokenization completed successfully.")
    return tokenized_features, tokenized_labels
