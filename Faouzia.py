import numpy as np

from abc import ABC
from utils.logger import logger
from utils.data_utils import labels_to_one_hot_map, preprocess_labels
from models.Hyperparameters import Hyperparameters
from models.Configuration import Configuration

# TODO: Update docstring
class Faouzia(ABC):
    """
    This class is used to determine the optimal configuration of parameters and hyperparameters for a multi-class classification 
    problem for tabular data only.

    This class does not include data preprocessing (e.g., dimensionality reduction, imputation, etc.). It is assumed that the data
    is already preprocessed and ready before passing to Faouzia.

    Attributes:

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

        #config = Configuration(hyperparameters=hyperparameters, weights=weights, biases=bias)

        return True

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
            activation_function_per_layer={1: 'relu'},
            learning_rate=0.01,
            num_epochs=1000,
            batch_size=32,
            optimizer='adam',
            loss_function='categorical_crossentropy',
            init_method='random_normal',
            regularization_method='l2'
        )

        logger.debug(f'Hyperparameters: {hyperparameters}')

        return hyperparameters
    
    # TODO: Update docstring
    def initialize_parameters(self, hyperparameters: Hyperparameters):
        """
        This method initializes the parameters (weights and bias) of the deep learning model.

        Parameters:
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.

        Returns:
            tuple(np.ndarray, np.ndarray): Weights and biases of the deep learning model.
        """

        logger.info("Initializing parameters...")

        if hyperparameters.init_method == 'random_normal':
            weights, bias = self.initialize_parameters_random_normal(hyperparameters)

        logger.debug(f'Weights: {weights}')
        logger.debug(f'Bias: {bias}')

        return weights, bias

    # TODO: Move to utils?
    def initialize_parameters_random_normal(self, hyperparameters: Hyperparameters):
        """
        This method initializes the parameters (weights and bias) of the deep learning model using the random normal method.

        Parameters:
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.

        Returns:
            tuple(np.ndarray, np.ndarray): Weights and biases of the deep learning model.
        """

        logger.info("Initializing parameters using random normal method...")

        weights = []
        bias = []

        for i in range(1, hyperparameters.num_hidden_layers + 1):
            weights.append(np.random.randn(hyperparameters.num_input_dimensions, hyperparameters.num_nodes_per_hidden_layer[i]))
            bias.append(np.zeros((1, hyperparameters.num_nodes_per_hidden_layer[i])))

        # TODO: Complete this method

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