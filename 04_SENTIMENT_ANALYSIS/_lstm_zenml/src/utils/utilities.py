import numpy
import pandas
import logging
from typing import Any


def sigmoid_function(x: Any) -> Any:
    """
    Description of sigmoid_function

    Args:
        x (Any):

    Returns:
        Any

    """
    # logging.info(f"using Sigmoid function for [{x}]")
    return 1 / (1 + numpy.exp(-x))


def stable_sigmoid_function(x: Any) -> Any:
    """
    Description of stable_sigmoid_function: as for large positive and negative
    values (exp(-x)) can overflow

    Args:
        x (Any):

    Returns:
        Any

    """
    return numpy.where(
        x >= 0, 1 / (1 + numpy.exp(-x)), numpy.exp(x) / (1 + numpy.exp(x))
    )


def d_sigmoid_function(x: Any) -> Any:
    """
    d_sigmoid_function: derivative of sigma(x)

    d(sigma)/dx = (sigma(x)) * (1 - sigma(x))

    Args:
        x (Any):

    Returns:
        Any

    """
    sigma_x = stable_sigmoid_function(x)
    return sigma_x * (1 - sigma_x)


def tanh_function(x: Any) -> Any:
    """
    Description of tanh_function

    Args:
        x (Any):

    Returns:
        Any

    """
    # logging.info(f"using tanh function for [{x}]")
    ex = numpy.exp(x)
    enx = numpy.exp(-x)
    return (ex - enx) / (ex + enx)
    # return 2 / (1 - numpy.exp(-(2 * x)))


def d_tanh_function(x: Any) -> Any:
    """
    d_tanh_function: derivative of tanh(x)

    d (tanh)/dx = 1 - (tanh(x) * tanh(x))

    Args:
        x (Any):

    Returns:
        Any

    """
    tanhx = tanh_function(x)
    return 1 - (tanhx**2)
