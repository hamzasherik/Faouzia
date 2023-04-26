import numpy as np
import copy

from typing import Dict
from abc import ABC
from utils.logger import logger
from utils.data_utils import labels_to_one_hot_map, preprocess_labels
from utils.activation_functions import relu, softmax
from models.Hyperparameters import Hyperparameters
from models.Configuration import Configuration


class Faouzia(ABC):
    """
    This class is used to determine the optimal configuration of parameters and hyperparameters for a multi-class classification 
    problem for tabular data only.

    This class does not include data preprocessing (e.g., dimensionality reduction, imputation, etc.). It is assumed that the data
    is already preprocessed and ready before passing to Faouzia.

    Attributes:
        features (np.ndarray): Features of the dataset.
        labels (np.ndarray): Labels of the dataset.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Constructs a new Faouzia object.

        Parameters:
            features (np.ndarray): Features of the dataset.
            labels (np.ndarray): Labels of the dataset.
        """
        
        logger.info("Initializing Faouzia object...")

        self.features = features
        self.labels = labels

        # TODO: Add validator that confirms contents of labels are one-hot encoded

    # TODO: Update docstring
    def execute_lifecycle(self) -> bool:
        """
        This method executes the lifecycle that determines the optimal configuration of parameters and hyperparameters for a deep 
        learning model in the following steps:
            - Data preprocessing
            -

        Returns:
            bool: True if lifecycle was executed successfully, False otherwise.
        """

        logger.info("Executing lifecycle...")

        hyperparameters = self.initialize_hyperparameters()
        weights, bias = self.initialize_parameters(hyperparameters)
        weights_trained, bias_trained = self.train_model(hyperparameters, weights, bias)

        #config = Configuration(hyperparameters=hyperparameters, weights=weights, biases=bias, accuracy=accuracy)

        return True

    def train_model(self, hyperparameters: Hyperparameters, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
        This method trains a deep learning model using the given hyperparameters, weights and bias.

        Parameters:
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.
            weights (np.ndarray): Weights of the deep learning model.
            bias (np.ndarray): Bias of the deep learning model.

        Returns:
            np.ndarray: Trained weights of the deep learning model.
            np.ndarray: Trained bias of the deep learning model.
        """

        logger.info("Training model...")

        weights_trained = copy.deepcopy(weights)
        bias_trained = copy.deepcopy(bias)

        forward_pass_output = self.forward_pass(self.features, weights, bias, hyperparameters)
        # loss = self.calculate_loss(self.labels, forward_pass_output)
        # gradients = self.backward_pass(self.features, self.labels, forward_pass_output, weights, bias)
        # weights_trained, bias_trained = self.update_parameters(weights, bias, gradients, hyperparameters)


    def forward_pass(self, features: np.ndarray, weights: np.ndarray, bias: np.ndarray, hyperparameters: Hyperparameters) -> np.ndarray:
        """
        This method performs a forward pass on the deep learning model.

        Parameters:
            features (np.ndarray): Features of the dataset.
            weights (np.ndarray): Weights of the deep learning model.
            bias (np.ndarray): Bias of the deep learning model.
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.

        Returns:
            np.ndarray: Output of the forward pass.
        """

        logger.info("Performing forward pass...")

        for layer in range(1, hyperparameters.num_hidden_layers + 2):
            # Input layer
            if layer == 1:
                weighted_sum = np.dot(features, weights[layer - 1]) + bias[layer - 1]

                activation = eval(hyperparameters.activation_function_per_layer[layer])(weighted_sum)

            # Hidden layers and output layer
            else:
                weighted_sum = np.dot(activation, weights[layer - 1]) + bias[layer - 1]

                activation = eval(hyperparameters.activation_function_per_layer[layer])(weighted_sum)

        logger.debug(f'Output of forward pass: {activation}')
        logger.debug(f'Output of forward pass shape: {activation.shape}')

        return activation
    
    def calculate_loss(self, labels: np.ndarray, forward_pass_output: np.ndarray, hyperparameters: Hyperparameters) -> float:
        """
        This method calculates the loss of the deep learning model.

        Parameters:
            labels (np.ndarray): Labels of the dataset.
            forward_pass_output (np.ndarray): Output of the forward pass.
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.

        Returns:
            float: Loss of the deep learning model.
        """

        logger.info("Calculating loss...")

        loss = eval(hyperparameters.loss_function)(labels, forward_pass_output)

        logger.debug(f'Loss: {loss}')

        return loss


    # TODO: Move to utils
    def initialize_hyperparameters(self) -> Hyperparameters:
        """
        This method initializes the hyperparameters of the deep learning model.

        Returns:
            Hyperparameters: Hyperparameters of the deep learning model.
        """

        logger.info("Initializing hyperparameters...")

        hyperparameters = Hyperparameters(
            num_input_dimensions=self.features.shape[1],
            num_output_nodes=self.labels.shape[1],
            num_hidden_layers=1,
            num_nodes_per_hidden_layer={1: 10},
            activation_function_per_layer={1: 'relu', 2: 'softmax'},
            learning_rate=0.01,
            num_epochs=1000,
            batch_size=32,
            optimizer='adam',
            loss_function='categorical_cross_entropy',
            init_method='random_normal',
            regularization_method='l2'
        )

        logger.debug(f'Hyperparameters: {hyperparameters}')

        return hyperparameters
    
    # TODO: Move to utils
    def initialize_parameters(self, hyperparameters: Hyperparameters) -> np.ndarray:
        """
        This method initializes the parameters (weights and bias) of the deep learning model.

        Parameters:
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.

        Returns:
            np.ndarray: Weights of the deep learning model.
            np.ndarray: Bias of the deep learning model.
        """

        logger.info("Initializing parameters...")

        if hyperparameters.init_method == 'random_normal':
            weights, bias = self.initialize_parameters_random_normal(hyperparameters)

        logger.debug(f'Weights: {weights}')
        logger.debug(f'Bias: {bias}')

        # TODO: debug log shape of weights and bias

        return weights, bias

    # TODO: Move to utils
    def initialize_parameters_random_normal(self, hyperparameters: Hyperparameters) -> np.ndarray:
        """
        This method initializes the parameters (weights and bias) of the deep learning model using the random normal method.

        Parameters:
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.

        Returns:
            np.ndarray: Weights of the deep learning model.
            np.ndarray: Bias of the deep learning model.
        """

        logger.info("Initializing parameters using random normal method...")

        weights = []
        bias = []

        # Input layer
        weights.append(np.random.randn(hyperparameters.num_input_dimensions, hyperparameters.num_nodes_per_hidden_layer[1]))
        bias.append(np.full((1, hyperparameters.num_nodes_per_hidden_layer[1]), 0.1))

        # Hidden layers
        for layer in range(2, hyperparameters.num_hidden_layers + 1):
            weights.append(np.random.randn(hyperparameters.num_nodes_per_hidden_layer[layer - 1], hyperparameters.num_nodes_per_hidden_layer[layer]))
            bias.append(np.full((1, hyperparameters.num_nodes_per_hidden_layer[layer]), 0.1))

        # Output layer
        weights.append(np.random.randn(hyperparameters.num_nodes_per_hidden_layer[hyperparameters.num_hidden_layers], hyperparameters.num_output_nodes))
        bias.append(np.full((1, hyperparameters.num_output_nodes), 0.1))

        return weights, bias





import pandas as pd

if __name__ == "__main__":
    test_data = pd.read_csv("IRIS.csv")
    test_data = test_data.values

    labels_map = labels_to_one_hot_map(test_data, 4)
    labels = preprocess_labels(test_data, 4, labels_map)

    features = np.delete(test_data, 4, axis=1)

    faouzia = Faouzia(features=features, labels=labels)
    faouzia.execute_lifecycle()
