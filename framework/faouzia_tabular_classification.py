import numpy as np

from typing import Dict
from abc import ABC
from utils.logger import logger
from utils.data_utils import labels_to_one_hot_map, preprocess_labels
from utils.activation_functions import softmax, relu, softmax_derivative, relu_derivative
from utils.loss_functions import categorical_cross_entropy, categorical_cross_entropy_derivative
from models.hyperparameters import Hyperparameters
from models.configuration import Configuration


class FaouziaTabularClassification(ABC):
    """
    This class is used to determine the optimal configuration of parameters and hyperparameters for a multi-class classification 
    problem for tabular data only.

    This class does not include data preprocessing (e.g., dimensionality reduction, imputation, etc.). It is assumed that the data
    is already preprocessed and ready before passing to Faouzia.

    Attributes:
        features (np.ndarray): Features of the dataset.
        labels (np.ndarray): Labels of the dataset.
    """

    # TODO: Remove constructor 
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

    # TODO: Remove this method from concrete implementation
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
        current_config = self.train_model(hyperparameters, weights, bias)

        logger.debug(f'Final configuration: {current_config}')

        return True

    def train_model(self, hyperparameters: Hyperparameters, weights: np.ndarray, bias: np.ndarray) -> Configuration:
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

        forward_pass_output = self.forward_pass(self.features, weights, bias, hyperparameters)
        loss = self.calculate_loss(self.labels, forward_pass_output, hyperparameters)
        weights_updated, bias_updated = self.backpropagation(hyperparameters, weights, bias, self.features, self.labels)

        epoch_ctr = 0

        while epoch_ctr < hyperparameters.num_epochs:

            forward_pass_output = self.forward_pass(self.features, weights_updated, bias_updated, hyperparameters)
            loss = self.calculate_loss(self.labels, forward_pass_output, hyperparameters)
            weights_updated, bias_updated = self.backpropagation(hyperparameters, weights_updated, bias_updated, self.features, self.labels)

        config = Configuration(hyperparameters=hyperparameters, weights=weights, biases=bias, loss=loss)

        logger.debug(f'Current configuration: {config}')

        return config

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

        for layer in range(1, len(hyperparameters.num_nodes_per_layer)):
            # Input layer
            if layer == 1:
                weighted_sum = np.dot(features, weights[layer - 1]) + bias[layer - 1]

                activation = eval(hyperparameters.activation_function_per_layer[layer])(weighted_sum)

            # Hidden layers and output layer
            else:
                weighted_sum = np.dot(activation, weights[layer - 1]) + bias[layer - 1]

                activation = eval(hyperparameters.activation_function_per_layer[layer])(weighted_sum)

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

    def backpropagation(self, hyperparameters: Hyperparameters, weights: np.ndarray, bias: np.ndarray, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        This method performs backpropagation on the deep learning model.

        Parameters:
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.
            weights (np.ndarray): Weights of the deep learning model.
            bias (np.ndarray): Bias of the deep learning model.
            features (np.ndarray): Features of the dataset.

        Returns:
            np.ndarray: Updated weights and bias of the deep learning model.
        """

        logger.info("Performing backpropagation...")

        weights_updated = weights.copy()
        bias_updated = bias.copy()

        logger.info("Performing forward pass...")

        zs = []
        activations = [features]

        for layer in range(1, len(hyperparameters.num_nodes_per_layer)):

            zs.append(np.dot(activations[layer - 1], weights[layer - 1]) + bias[layer - 1])
            activations.append(eval(hyperparameters.activation_function_per_layer[layer])(zs[layer - 1]))

        logger.info("Performing backward pass...")

        delta = []

        activation_function_per_layer_copy = hyperparameters.activation_function_per_layer.copy() 

        key, output_layer_activation_function = activation_function_per_layer_copy.popitem()
  
        # Output layer
        delta.append(eval(hyperparameters.loss_function + '_derivative')(labels, activations[-1]) * eval(output_layer_activation_function + '_derivative')(zs[-1]))

        # Hidden layers
        for layer in range((len(hyperparameters.num_nodes_per_layer) - 2), 0, -1):
            delta.append(np.dot(delta[-1], weights[layer].T) * eval(hyperparameters.activation_function_per_layer[layer] + '_derivative')(zs[layer - 1]))

        delta.reverse()

        logger.info("Updating weights and bias...")

        for layer in range(1, len(hyperparameters.num_nodes_per_layer)):

            weights_updated[layer - 1] = weights_updated[layer - 1] - (hyperparameters.learning_rate * np.sum(np.dot(activations[layer].T, delta[layer - 1])))
            bias_updated[layer - 1] = bias_updated[layer - 1] - (hyperparameters.learning_rate * np.sum(delta[layer - 1], axis=0, keepdims=True))

        logger.debug(f'Adjustment to weights: {(hyperparameters.learning_rate * np.sum(np.dot(activations[layer].T, delta[layer - 1])))}')
        logger.debug(f'Adjustment to bias: {(hyperparameters.learning_rate * np.sum(delta[layer - 1], axis=0, keepdims=True))}')

        return weights_updated, bias_updated

    def initialize_hyperparameters(self) -> Hyperparameters:
        """
        This method initializes the hyperparameters of the deep learning model.

        Returns:
            Hyperparameters: Hyperparameters of the deep learning model.
        """

        logger.info("Initializing hyperparameters...")

        hyperparameters = Hyperparameters(
            num_nodes_per_layer={1: self.features.shape[1], 2: self.labels.shape[1]},
            activation_function_per_layer={1: 'relu', 2: 'softmax'},
            learning_rate=0.001,
            num_epochs=1000,
            batch_size=32,
            optimizer='adam',
            loss_function='categorical_cross_entropy',
            init_method='random_normal',
            regularization_method='l2'
        )

        logger.debug(f'Hyperparameters: {hyperparameters}')

        return hyperparameters

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

        for key, value in hyperparameters.num_nodes_per_layer.items():

            if key == len(hyperparameters.num_nodes_per_layer):
                continue
            
            weights.append(np.random.randn(value, hyperparameters.num_nodes_per_layer[key + 1]))
            bias.append(np.full((1, hyperparameters.num_nodes_per_layer[key + 1]), 0.1))

        return weights, bias