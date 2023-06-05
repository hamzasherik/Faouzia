import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from models.model_config import ModelConfig
from utils.logger import logger


class Faouzia(ABC):
    """
    This is the Faouzia base class which will support concrete implementations for binary and multi-class classification
    tasks
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Constructs a new Faouzia object.

        Parameters:
            data: Dataframe containing the dataset.
        """

        self.data = data

    def execute_lifecycle(self) -> None:
        """
        This method executes the lifecycle which starts off with preprocessing data, then selecting the model, then
        training the model, then finally evaluating the model.
        """

        logger.info('Executing lifecycle...')

        data_preprocessed = self.preprocess_data()
        model_config = self.model_selection(data_preprocessed)
        self.train_model(model_config)
        accuracy = self.evaluate_model(model_config)

        return

    # NOTE: preprocess data in pandas then convert to numpy for model training
    @abstractmethod
    def preprocess_data(self) -> np.ndarray:
        """
        This method preprocesses the dataset according to user specifications.

        The preprocessing steps will vary depending on the concrete implementation and the data.

        Returns:
            np.ndarray: Preprocessed dataset.
        """

        pass

    @abstractmethod
    def model_selection(self, data: np.ndarray) -> ModelConfig:
        """
        This method selects the model to be used for training and evaluation.

        This method includes hyperparameter selection, architecture selection, hyperparameter tuning,
        etc. The specific steps will vary depending on the concrete implementation and the data.

        Parameters:
            data: Preprocessed data that will be used to select the model

        Returns:
            ModelConfig: Model configuration containing the hyperparameters, architecture, etc. of the deep
            learning model.
        """

        pass

    @abstractmethod
    def train_model(self, model_config: ModelConfig) -> np.ndarray:
        """
        This method trains a deep learning model.

        Parameters:
            model_config: Configuration (including hyperparameters) of the deep learning model.
        """

        pass

    @abstractmethod
    def evaluate_model(self, model_config: ModelConfig) -> float:
        """
        This method evaluates a deep learning model.

        Parameters:
            model_config: Configuration (including hyperparameters) of the deep learning model.

        Returns:
            float: Evaluation of the deep learning model.
        """

        pass

    def forward_pass(self, features: np.ndarray, model_config: ModelConfig, weights: np.ndarray, bias: np.ndarray) -> \
            np.ndarray:
        """
        This concrete method applies the forward pass of an input dataset through a neural network

        Parameters:
                features: Features for all samples from input dataset
                model_config: Model configuration housing hyperparameters needed to complete forward pass
                weights: Weights of neural network
                bias: Bias of neural network

        Returns:
            np.ndarray: Numpy array housing output (prediction or probability of output node activations) for each
            sample
        """

        logger.info("Performing forward pass...")

        for layer in range(1, len(model_config.num_nodes_per_layer)):
            # Input layer
            if layer == 1:
                weighted_sum = np.dot(features, weights[layer - 1]) + bias[layer - 1]

                activation = eval(model_config.activation_function_per_layer[layer])(weighted_sum)

            # Hidden layers and output layer
            elif layer > 1:
                weighted_sum = np.dot(activation, weights[layer - 1]) + bias[layer - 1]

                activation = eval(model_config.activation_function_per_layer[layer])(weighted_sum)

        logger.debug(f'Output of forward pass shape: {activation.shape}')

        return activation

    def calculate_loss(self, labels: np.ndarray, forward_pass_output: np.ndarray, model_config: ModelConfig) -> float:
        """
        This method calculates the loss of the deep learning model.

        Parameters:
            labels: Labels of the dataset
            forward_pass_output: Output of the forward pass
            model_config: Model configuration housing hyperparameters needed to calculate loss

        Returns:
            float: Loss of the deep learning model.
        """

        logger.info("Calculating loss...")

        loss = eval(model_config.loss_function)(labels, forward_pass_output)

        logger.debug(f'Loss: {loss}')

        return loss

    # TODO: Need to clean-up this implementation of backprop.
    def backpropagation(self, model_config: ModelConfig, weights: np.ndarray, bias: np.ndarray,
                        features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        This method performs backpropagation on the deep learning model.

        Parameters:
            model_config: Model configuration housing hyperparameters needed for backpropagation
            weights: Weights of the deep learning model.
            bias: Bias of the deep learning model.
            features: Features of the dataset.

        Returns:
            np.ndarray: Updated weights and bias of the deep learning model.
        """

        logger.info("Performing backpropagation...")

        weights_updated = weights.copy()
        bias_updated = bias.copy()

        logger.info("Performing forward pass...")

        zs = []
        activations = [features]

        for layer in range(1, len(model_config.num_nodes_per_layer)):
            zs.append(np.dot(activations[layer - 1], weights[layer - 1]) + bias[layer - 1])
            activations.append(eval(model_config.activation_function_per_layer[layer])(zs[layer - 1]))

        logger.info("Performing backward pass...")

        delta = []

        activation_function_per_layer_copy = model_config.activation_function_per_layer.copy()

        key, output_layer_activation_function = activation_function_per_layer_copy.popitem()

        # Output layer
        delta.append(eval(model_config.loss_function + '_derivative')(labels, activations[-1]) * eval(
            output_layer_activation_function + '_derivative')(zs[-1]))

        # Hidden layers
        for layer in range((len(model_config.num_nodes_per_layer) - 2), 0, -1):
            delta.append(np.dot(delta[-1], weights[layer].T) * eval(
                model_config.activation_function_per_layer[layer] + '_derivative')(zs[layer - 1]))

        delta.reverse()

        logger.info("Updating weights and bias...")

        for layer in range(1, len(model_config.num_nodes_per_layer)):
            weights_updated[layer - 1] = weights_updated[layer - 1] - (
                        model_config.learning_rate * np.sum(np.dot(activations[layer].T, delta[layer - 1])))
            bias_updated[layer - 1] = bias_updated[layer - 1] - (
                        model_config.learning_rate * np.sum(delta[layer - 1], axis=0, keepdims=True))

        # TODO: remove these debug logs.
        logger.debug(
            f'Adjustment to weights: {(model_config.learning_rate * np.sum(np.dot(activations[layer].T, delta[layer - 1])))}')
        logger.debug(
            f'Adjustment to bias: {(model_config.learning_rate * np.sum(delta[layer - 1], axis=0, keepdims=True))}')

        # TODO: update type hint in signature to match return type
        return weights_updated, bias_updated