from pydantic import BaseModel, validator
from typing import Dict


class Hyperparameters(BaseModel):
    """
    This class is used to store the hyperparameters of a deep learning model for Faouzia. It also includes both built-in and custom data validations.

    Attributes:
        num_input_dimensions (int): Number of input dimensions.
        num_output_nodes (int): Number of output nodes.
        num_hidden_layers (int): Number of hidden layers.
        num_nodes_per_hidden_layer (dict[int, int]): Number of nodes per hidden layer.
        activation_function_per_layer (dict[int, str]): Activation function per layer.
        learning_rate (float): Learning rate.
        optimizer (str): Optimizer.
        loss_function (str): Loss function.
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        init_method (str): Initialization method.
        regularization_method (str): Regularization method.
    """

    num_input_dimensions: int
    num_output_nodes: int
    num_hidden_layers: int
    num_nodes_per_hidden_layer: dict[int, int]  # TODO: use Dict class?
    activation_function_per_layer: dict[int, str]
    learning_rate: float
    optimizer: str
    loss_function: str
    num_epochs: int
    batch_size: int
    init_method: str
    regularization_method: str
