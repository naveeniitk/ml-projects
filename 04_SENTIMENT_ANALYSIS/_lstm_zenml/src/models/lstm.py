import torch
import numpy
import torch.nn as torchNeuralNetwork
from typing import Tuple, Dict, Any, Annotated
from utils.utilities import sigmoid_function, tanh_function


class LstmClassifier(torchNeuralNetwork.Module):
    """
    LSTM-based text classification model: single-layer LSTM using numpy

    Parameters are:
        Wx: (input_size, 4 * hidden_size)
        Wh: (hidden_size, 4 * hidden_size)
        b : (4 * hidden_size,)

    Inheritance:
        torchNeuralNetwork.Module: Base class for all neural network modules.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        seed: int = 0,
        # *args: Tuple[Any],
        # **kwargs: Dict[str, Any]
    ):
        """
        Description of __init__

        Args:
            self (undefined):
            input_size (int):
            hidden_size (int):
                hyperparameter that determines the dimensionality of the LSTM’s hidden state

                defines the size of the internal weight matrices

            seed (int=0):
        """

        # random generator object based on the Mersenne Twister algorithm
        random_number_generator = numpy.random.RandomState(seed=seed)

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier(Glorot) initialization is a popular technique that helps
        # maintain a stable variance of activations and gradients across
        # layers, addressing issues like vanishing or exploding gradients
        standard_deviation = 1.0 / numpy.sqrt(hidden_size + input_size)

        self.Wx = random_number_generator.normal(
            scale=standard_deviation, size=(input_size, 4 * hidden_size)
        )

        self.Wh = random_number_generator.normal(
            scale=standard_deviation, size=(hidden_size, 4 * hidden_size)
        )

        self.b = numpy.zeros(4 * hidden_size)

        # derivatives of Wx, Wh, b
        derivative_Wx = numpy.zeros_like(self.Wx)
        derivative_Wh = numpy.zeros_like(self.Wh)
        derivative_b = numpy.zeros_like(self.b)

    def init_lstm_state(self, batch_size: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Init hidden state and cell state for the model

        Args:
            self (undefined):
            batch_size (int):

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        shape_of_state: Tuple[int, int] = (batch_size, self.hidden_size)

        # short-term representation of the input sequence
        hidden_state_0 = numpy.zeros(shape=shape_of_state)

        # long-term memory storage that helps mitigate vanishing gradients
        cell_state_0 = numpy.zeros(shape=shape_of_state)

        return hidden_state_0, cell_state_0

    def forward_step(
        self,
        input_data: numpy.ndarray,
        prev_hidden_state: numpy.ndarray,
        prev_cell_state: numpy.ndarray,
    ) -> Tuple[
        numpy.ndarray,
        numpy.ndarray,
        Tuple,
    ]:
        """
        one forward step of the lstm cell

        Args:
            self (undefined):
            input_data (numpy.ndarray):
            prev_hidden_state (numpy.ndarray):
            prev_cell_state (numpy.ndarray):

        Returns:
            Tuple[ numpy.ndarray, numpy.ndarray, Tuple, ]
            new_hidden_state, new_cell_state, state_cache
        """
        # (batch, 4H)
        A = (input_data @ self.Wx) + (prev_hidden_state @ self.Wh) + (self.b)

        H = self.hidden_size
        input_gate = sigmoid_function(A[:, 0 * H : 1 * H])  # i
        forget_gate = sigmoid_function(A[:, 1 * H : 2 * H])  # f
        output_gate = sigmoid_function(A[:, 2 * H : 3 * H])  # o
        candidate_gate = tanh_function(A[:, 3 * H : 4 * H])  # g

        # c_t = f * c_prev + i * g
        new_cell_state = (forget_gate * prev_cell_state) + (input_gate * candidate_gate)

        # h_t = o * tanh(c_t)
        new_hidden_state = output_gate * tanh_function(new_cell_state)

        state_cache = (
            input_data,
            prev_hidden_state,
            prev_cell_state,
            input_gate,
            forget_gate,
            output_gate,
            candidate_gate,
            new_cell_state,
            H,  # current hidden size
            A,
        )

        return new_hidden_state, new_cell_state, state_cache
