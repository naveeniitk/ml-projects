import numpy
from zenml.steps import step
from typing import List, Dict, Tuple, Annotated
from config.params import MAX_VOCAB_SIZE, MAX_LENGTH


def embedding_using_vocabulary_building(
    tokenized_features: List[str],
    max_vocab_size: int = MAX_VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
) -> Dict[str, int]:
    """
    Embeds the input texts using a vocabulary built from the texts.

    Args:
        texts: List of input strings.
        max_vocab_size: Maximum size of the vocabulary.
        max_length: Fixed length for the output sequences.

    Returns:
        Dictionary mapping tokens to integer indices.
    """
    frequency_dict: Dict[str, int] = {}
    for feature_tokens in tokenized_features:
        for token in feature_tokens:
            frequency_dict[token] = frequency_dict.get(token, 0) + 1

    sorted_words_via_freq = sorted(
        frequency_dict.items(), key=lambda x: x[1], reverse=True
    )

    top_words = sorted_words_via_freq[:max_vocab_size]

    # Reserve special tokens
    word_to_index = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for index, (word, _) in enumerate(top_words, start=2):
        word_to_index[word] = index

    return word_to_index


def encode_vocabulary_embedding(
    tokenized_texts: List[List[str]],
    word2idx: Dict[str, int],
    max_seq_len: int = MAX_LENGTH,
) -> numpy.ndarray:
    """
    Convert tokenized texts into sequences of indices based on the vocabulary.

    Args:
        tokenized_texts: List of tokenized input strings.
        word2idx: Vocabulary mapping tokens to indices.
        max_seq_len: Fixed length for the output sequences.

    Returns:
        Numpy array of shape (num_texts, max_seq_len) with token indices.
    """
    encoded_sequences = []
    for tokens in tokenized_texts:
        sequence = [
            word2idx.get(token, word2idx["<unk>"]) for token in tokens[:max_seq_len]
        ]
        sequence += [word2idx["<pad>"]] * (max_seq_len - len(sequence))
        encoded_sequences.append(sequence)

    return numpy.ndarray(encoded_sequences)
