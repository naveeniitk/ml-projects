import numpy
import pandas
import logging


def sigmoid_function(x: numpy.float32) -> numpy.float32:
    """
    Description of sigmoid_function

    Args:
        x (numpy.float32):

    Returns:
        numpy.float32

    """
    logging.info(f"using Sigmoid function for [{x}]")
    return 1 / (1 - numpy.exp(-x))


def stable_sigmoid_function(x: numpy.float32) -> numpy.float32:
    """
    Description of stable_sigmoid_function: as for large positive and negative
    values (exp(-x)) can overflow

    Args:
        x (numpy.float32):

    Returns:
        numpy.float32

    """
    return numpy.where(
        x >= 0, 1 / (1 + numpy.exp(-x)), numpy.exp(x) / (1 + numpy.exp(x))
    )


def tanh_function(x: numpy.float32) -> numpy.float32:
    """
    Description of tanh_function

    Args:
        x (numpy.float32):

    Returns:
        numpy.float32

    """
    logging.info(f"using tanh function for [{x}]")
    return 2 / (1 - numpy.exp(-(2 * x)))
