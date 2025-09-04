import torch
import numpy
import torch.nn as torchNeuralNetwork
from typing import Tuple, Dict, Any, Annotated
from utils.utilities import (
    sigmoid_function,
    tanh_function,
    d_tanh_function,
    d_sigmoid_function,
)
import config.params as config_params


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
        learning_rate: int = config_params.LSTM_LEARNING_RATE,
        # *args: Tuple[Any],
        # **kwargs: Dict[str, Any]
    ):
        # inherit methods from torchNeuralNetwork.Module
        super().__init__()

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
        self.learning_rate = learning_rate

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

        self.output_weights = random_number_generator.normal(
            scale=standard_deviation,
            size=(self.hidden_size, config_params.NUMBER_CLASSES),
        )
        self.output_bias = random_number_generator.normal(
            scale=standard_deviation, size=(config_params.NUMBER_CLASSES)
        )

        # derivatives of Wx, Wh, b
        self.derivative_Wx = numpy.zeros_like(self.Wx)
        self.derivative_Wh = numpy.zeros_like(self.Wh)
        self.derivative_b = numpy.zeros_like(self.b)

        self.derivative_output_weights = numpy.zeros_like(self.output_weights)
        self.derivative_output_bias = numpy.zeros_like(self.output_bias)

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

    def forward(
        self,
        X: numpy.ndarray,
        hidden_state_0: numpy.ndarray = None,
        cell_state_0: numpy.ndarray = None,
    ) -> Tuple[
        Annotated[numpy.ndarray, "final_hidden_state"],
        Annotated[numpy.ndarray, "final_cell_state"],
        Annotated[numpy.ndarray, "final_cache_state"],
        Tuple[
            Annotated[numpy.ndarray, "hidden_state_0"],
            Annotated[numpy.ndarray, "cell_state_0"],
        ],
    ]:
        """
        forward process using forward

        Args:
            self (undefined):
            X (numpy.ndarray):
            hidden_state_0 (numpy.ndarray|None=None):
            cell_state_0 (numpy.ndarray|None=None):

        Returns:
            Tuple[        Annotated[numpy.ndarray, "final_hidden_state"],        Annotated[numpy.ndarray, "final_cell_state"],        Annotated[numpy.ndarray, "final_cache_state"],        Tuple[            Annotated[numpy.ndarray, "hidden_state_0"],            Annotated[numpy.ndarray, "cell_state_0"],        ],    ]

        """
        # T: number_of_time_steps
        T, batch = len(X), 1

        if hidden_state_0 is None or cell_state_0 is None:
            hidden_state_0, cell_state_0 = self.init_lstm_state(batch_size=batch)

        final_hidden_state = numpy.zeros((T, batch, self.hidden_size))
        final_cell_state = numpy.zeros((T, batch, self.hidden_size))
        final_cache_state = [None] * T

        prev_hidden_state, prev_cell_state = hidden_state_0, cell_state_0

        for t in range(T):
            current_input_x_t = X[t]
            current_input_x_t = X[t].reshape(1, -1)

            new_hidden_state, new_cell_state, new_state_cache = self.forward_step(
                input_data=current_input_x_t,
                prev_hidden_state=prev_hidden_state,
                prev_cell_state=prev_cell_state,
            )

            final_hidden_state[t] = new_hidden_state
            final_cell_state[t] = new_cell_state
            final_cache_state[t] = new_state_cache

        return (
            final_hidden_state,
            final_cell_state,
            final_cache_state,
            (hidden_state_0, cell_state_0),
        )

    def backward_step(
        self,
        next_derivative_of_hidden_state: Any,
        next_derivative_of_cell_state: Any,
        current_cache_state: Any,
    ) -> Tuple[
        Annotated[numpy.ndarray, "derivative_input"],
        Annotated[numpy.ndarray, "derivative_prev_hidden_state"],
        Annotated[numpy.ndarray, "derivative_prev_cell_state"],
    ]:
        """
        backward_step

        Args:
            self (undefined):
            next_derivative_of_hidden_state (Any):
            next_derivative_of_cell_state (Any):
            current_cache_state (Any):

        Returns:
            Tuple[        Annotated[numpy.ndarray, "derivative_input"],        Annotated[numpy.ndarray, "derivative_prev_hidden_state"],        Annotated[numpy.ndarray, "derivative_prev_cell_state"],    ]

        """

        (
            current_input,
            prev_hidden_state,
            prev_cell_state,
            input_gate,
            forget_gate,
            output_gate,
            candidate_gate,
            cur_cell_state,
            A,
            _,
        ) = current_cache_state

        H = self.hidden_size

        # derivatives calculation
        derivative_output_gate = next_derivative_of_hidden_state * tanh_function(
            cur_cell_state
        )
        derivative_cell_state = (
            next_derivative_of_hidden_state
            * output_gate
            * d_tanh_function(cur_cell_state)
            + next_derivative_of_cell_state
        )

        derivative_input_gate = derivative_cell_state * candidate_gate
        derivative_candidate_gate = derivative_cell_state * input_gate
        derivative_forget_gate = derivative_cell_state * prev_cell_state

        derivative_prev_cell_state = derivative_cell_state * forget_gate

        # gate activation derivatives
        derivative_A_input_gate = derivative_input_gate * d_sigmoid_function(input_gate)
        derivative_A_forget_gate = derivative_forget_gate * d_sigmoid_function(
            forget_gate
        )
        derivative_A_output_gate = derivative_output_gate * d_sigmoid_function(
            output_gate
        )
        derivative_A_candidate_gate = derivative_candidate_gate * d_sigmoid_function(
            candidate_gate
        )

        derivative_A = numpy.concatenate(
            [
                derivative_A_input_gate,
                derivative_A_forget_gate,
                derivative_A_output_gate,
                derivative_A_candidate_gate,
            ],
            axis=1,
        )

        # derivative wrt params
        self.derivative_Wx += (current_input.T) @ derivative_A
        self.derivative_Wh += prev_hidden_state.T @ derivative_A
        self.derivative_b += derivative_A.sum(axis=0)

        derivative_input = derivative_A @ self.Wx.T
        derivative_prev_hidden_state = derivative_A @ self.Wh.T

        return (
            derivative_input,
            derivative_prev_hidden_state,
            derivative_prev_cell_state,
        )

    def backward(
        self,
        derivative_loss_wrt_current_hidden_state: numpy.ndarray,
        cache_state: Any,
    ) -> Annotated[numpy.ndarray, "derivative_full_input_data"]:
        """
        backward

        Args:
            self (undefined):
            derivative_loss_wrt_current_hidden_state (numpy.ndarray):
            cache_state (Any):

        Returns:
            Annotated[numpy.ndarray, "derivative_full_input_data"]

        """
        T = len(cache_state)
        batch = 1
        # derivative_loss_wrt_current_hidden_state.shape[1]

        input_size = self.input_size
        derivative_full_input_data = numpy.zeros((T, batch, input_size))

        # resetting derivatives
        self.derivative_Wx.fill(0)
        self.derivative_Wh.fill(0)
        self.derivative_b.fill(0)

        self.derivative_output_bias.fill(0.0)
        self.derivative_output_weights.fill(0)

        next_derivative_of_hidden_state = numpy.zeros((batch, self.hidden_size))
        next_derivative_of_cell_state = numpy.zeros((batch, self.hidden_size))

        for t in reversed(range(T)):
            total_derivative_hidden_state = (
                derivative_loss_wrt_current_hidden_state
                + next_derivative_of_hidden_state
            )

            (
                derivative_input_data,
                next_derivative_of_hidden_state,
                next_derivative_of_cell_state,
            ) = self.backward_step(
                next_derivative_of_hidden_state=total_derivative_hidden_state,
                next_derivative_of_cell_state=next_derivative_of_cell_state,
                current_cache_state=cache_state[t],
            )

            derivative_full_input_data[t] = derivative_input_data

        return derivative_full_input_data

    @property
    def get_params(self):
        return [self.Wx, self.Wh, self.b]

    @property
    def get_derivatives(self):
        return [self.derivative_Wh, self.derivative_Wh, self.derivative_b]

    def apply_derivatives(self):
        self.Wx -= self.learning_rate * self.derivative_Wx
        self.Wh -= self.learning_rate * self.derivative_Wh
        self.b -= self.learning_rate * self.derivative_b

        self.output_weights -= self.learning_rate * self.derivative_output_weights
        self.output_bias -= self.learning_rate * self.derivative_output_bias

    def sequence_forward(
        self,
        seq: numpy.ndarray,
    ) -> numpy.ndarray:
        return seq @ self.output_weights + self.output_bias

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        total_epochs: int,
    ) -> None:
        """
        Train the LSTM model using cross-entropy loss.

        Args:
            X (numpy.ndarray): training data of shape (num_samples, input_dim)
            y (numpy.ndarray): training labels of shape (num_samples,)
            model (LstmClassifier): the LSTM model
            total_epochs (int): number of training epochs

        Returns:
            None
        """
        total_samples = X.shape[0]

        for epoch in range(total_epochs):
            total_loss = 0.0
            for i in range(total_samples):
                x_i = X[i].reshape(1, -1)  # (1, seq_len) or (1, input_dim)
                y_true = y[i]  # scalar class index (int)
                # forward pass
                final_hidden_state, final_cell_state, cache, _ = self.forward(x_i)
                # last hidden state -> logits
                logits = self.sequence_forward(seq=final_hidden_state[-1])
                # logits shape: (num_classes,)
                # softmax
                exp_logits = numpy.exp(logits - numpy.max(logits))  # stability
                probs = exp_logits / numpy.sum(exp_logits)
                probs = probs.squeeze()
                # compute log loss
                loss = -numpy.log(probs[y_true] + 1e-12)
                total_loss += loss
                # gradient wrt logits (cross-entropy derivative)
                y_one_hot = numpy.zeros_like(probs)
                y_one_hot[y_true] = 1
                d_logits = probs - y_one_hot  # shape (num_classes,)

                weight_projection = d_logits @ self.output_weights.T

                # backprop through classifier head + LSTM
                self.backward(
                    derivative_loss_wrt_current_hidden_state=weight_projection,
                    cache_state=cache,
                )
                # update weights
                self.apply_derivatives()

            print(
                f"Epoch {epoch+1}/{total_epochs}, Loss: {total_loss/total_samples:.4f}"
            )

    def predict(
        self,
        X: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        X: (N, input_size) where N = number of data points
        Returns: (N, C) predictions
        """
        preds = []
        for i in range(X.shape[0]):  # 15 samples
            # Forward through RNN to get hidden representation
            h_T, _, _, _ = self.forward(X[i])  
            
            # Pass last hidden state to linear layer
            logits = self.sequence_forward(h_T[-1])
            
            # Convert to prediction
            if config_params.NUMBER_CLASSES == 2:
                probs = sigmoid_function(logits)
                # preds.append(int(probs >= 0.5))
                preds.append(probs)
            else:
                exp_logits = numpy.exp(logits - numpy.max(logits))
                probs = exp_logits / numpy.sum(exp_logits)
                preds.append(int(numpy.argmax(probs)))
        preds = numpy.array(preds)
        return preds
