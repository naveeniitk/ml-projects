import numpy
import pandas
import logging
from zenml.steps import step
from models.lstm import LstmClassifier
from config.params import HIDDEN_SIZE, LSTM_SEED, LSTM_TOTAL_EPOCHS
from typing import Any


@step(enable_cache=False)
def actual_training_step(
    X: numpy.ndarray,
    y: numpy.ndarray,
    model: LstmClassifier,
    total_epochs: int,
) -> LstmClassifier:
    """
    Train the LSTM model for multiple epochs.

    Args:
        X (numpy.ndarray): training data of shape (num_samples, input_dim)
        y (numpy.ndarray): training labels of shape (num_samples,)
        model (LstmClassifier): the LSTM model
        total_epochs (int): number of training epochs

    Returns:
        LstmClassifier
    """
    logging.info(f"Starting training")
    total_samples = X.shape[0]

    for epoch in range(total_epochs):
        total_loss = 0.0

        for i in range(total_samples):

            x_i = X[i].reshape(1, -1)  # shape (1, 384)
            y_true = y[i]

            # forward pass
            final_hidden_state, final_cell_state, cache, _ = model.forward(x_i)

            # use last hidden state as prediction
            y_pred = final_hidden_state[-1].squeeze()  # shape (hidden_size,)

            # for scalar prediction
            # print(f"hasAttr, why: {hasattr(model, "Why")}")

            # compute loss (MSE)
            loss = numpy.mean((y_pred - y_true) ** 2)
            total_loss += loss

            # backward pass
            derivative_loss = 2 * (y_pred - y_true) / y_true.size
            model.backward(
                derivative_loss_wrt_current_hidden_state=derivative_loss,
                cache_state=cache,
            )

            # update weights
            model.apply_derivatives()

        average_loss = total_loss / total_samples
        logging.info(f"Epoch {epoch + 1} / {total_epochs}, Loss: {average_loss:.6f}")

    return model


# @step(enable_cache=False)
# def actual_training_step(
#     X: numpy.ndarray,
#     y: numpy.ndarray,
#     model: LstmClassifier,
#     total_epochs: int,
# ) -> LstmClassifier:
#     """
#     actual_training_step

#     Args:
#         X (numpy.ndarray): training data
#         y (pandas.Series): training labels
#         model (LstmClassifier):
#         total_epochs: (int)

#     Returns:
#         LstmClassifier

#     """
#     logging.info(f"Starting training")
#     total_samples = X.shape[0]

#     loss_details = [0] * total_epochs
#     epoch_number = [0] * total_epochs

#     for epoch in range(total_epochs):
#         total_loss = 0
#         y_true = y

#         # forward pass
#         # final_hidden_state, final_cell_state, final_cache_state, hidden_state_0, cell_state_0 =
#         y_pred, cache, a, b = model.forward(X)


#         # computing loss
#         loss = numpy.mean((y_pred - y_true) ** 2)
#         total_loss += loss

#         # back propagation
#         derivative_loss = 2 * (y_pred - y_true) / y_true.size
#         model.backward(
#             derivative_loss_wrt_current_hidden_state=derivative_loss,
#             cache_state=cache,
#         )

#         # update weights
#         model.apply_derivatives()

#         average_loss = total_loss / total_samples
#         loss_details[epoch] = average_loss
#         epoch_number[epoch] = epoch + 1
#         logging.info(f"Epoch {epoch + 1} / {total_epochs}, Loss: {average_loss:.6f}")

#     return model


@step(enable_cache=False)
def trainer_step(
    X_train: numpy.ndarray,
    y_train: pandas.Series,
    hidden_size: int = HIDDEN_SIZE,
) -> LstmClassifier:

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    y_train = y_train.to_numpy()

    logging.info("Initializing the LSTM model...")
    lstm_model: Any = LstmClassifier(
        input_size=len(X_train[0]), hidden_size=hidden_size, seed=LSTM_SEED
    )

    logging.info(f"Starting LSTM training...")
    lstm_model = actual_training_step(
        X=X_train, y=y_train, model=lstm_model, total_epochs=LSTM_TOTAL_EPOCHS
    )

    logging.info("Training completed successfully.")
    return lstm_model
