import numpy as np
import pandas as pd

from typing import Optional
from abc import ABC
from utils.logger import logger
from utils.data_utils import labels_to_one_hot_map, preprocess_labels
from models.Hyperparameters import Hyperparameters


class Faouzia(ABC):
    """
    This class is used to determine the optimal configuration of parameters and hyperparameters for a multi-class classification 
    problem for tabular data only.

    This class does not include data preprocessing (e.g., dimensionality reduction, imputation, etc.). It is assumed that the data
    is already preprocessed and ready before passing to Faouzia.

    Attributes:
        raw_data (np.ndarray): Raw tabular data stored in numpy array.
        data_copy (np.ndarray): Copy of raw_data.
        labels_col_idx (int): Index of the labels column.
        hyperparameters (dict): Dictionary of hyperparameters.
        weights (np.ndarray): Weights of the deep learning model.
        bias (np.ndarray): Bias of the deep learning model.
        labels_map (dict): Dictionary that maps labels to one-hot vectors.
    """

    # TODO: Update docstring
    def __init__(self, data: pd.DataFrame, labels_column_index: int) -> None:
        """
        Constructs a new Faouzia object.

        Parameters:
            data (pd.DataFrame): Raw tabular data stored in pandas dataframe.
            labels_col_idx (int): Index of the labels column.
        """
        
        logger.info("Initializing Faouzia object...")

        if not isinstance(data, pd.DataFrame):
            logger.error(f'{type(data)} is an invalid datatype. Input data must be a pandas DataFrame')
            raise TypeError(f'{type(data)} is an invalid datatype. Input data must be a pandas DataFrame')

        self.labels_column_index = labels_column_index
        self.raw_data = data.values

        # NOTE: I could require input data to be preprocessed where labels are one-hot encoded. This would simplify the code.
        # TODO: Require input data to be preprocessed where labels are one-hot encoded. Another library I develop will be used to
        # preprocess data.
        self.labels_map = labels_to_one_hot_map(self.data_copy, self.labels_column_index)
        self.data_processed = preprocess_labels(self.raw_data, self.labels_column_index, self.labels_map)


        # TODO: Create pydantic model that stores weights, bias, and hyperparameters (not in the constructor). This model will be used
        # internally to store configuration and their corresponding accuracies before determining the optimal configuration. The 
        # optimal configuration can be accessed by a user through a getter method or directly through the Faouzia object.


        self.hyperparameters = {}  # TODO: Move Hyperparameters initialization to constructor
        self.weights = []  # TODO: Move weights initialization to constructor
        self.bias = []  # TODO: Move bias initialization to constructor

        # NOTE: For each of the 3 attributes above, we'll need to locally store their values and corresponding accuracy after each
        # iteration of the lifecycle. This will allow us to determine the optimal configuration of parameters and hyperparameters.
        # This means we may need to create a model class that stores the weights, bias, and hyperparameters.


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

        self.initialize_hyperparameters()

        
    def initialize_hyperparameters(self) -> None:
        """
        This method initializes the hyperparameters of the deep learning model and applies those values to the object's 
        hyperparameters attribute.
        """

        logger.info("Initializing hyperparameters...")
        # TODO: Add output layer activation function to activation_function_per_layer
        self.hyperparameters = Hyperparameters(
            num_input_dimensions=self.data_copy.shape[1] - 1,
            num_output_nodes=len(self.labels_map),
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

        logger.debug(f'Hyperparameters: {self.hyperparameters}')

    
    def initialize_parameters(self) -> bool:
        """
        This method initializes the parameters of the deep learning model and applies those values to the object's weights and bias
        attributes.

        Returns:
            bool: True if parameters were initialized successfully, False otherwise.
        """

        logger.info("Initializing parameters...")

        self.weights.append(np.random.randn(self.hyperparameters.num_input_dimensions, 
                                            self.hyperparameters.num_nodes_per_hidden_layer[1]))
        self.bias.append(np.zeros((1, self.hyperparameters.num_nodes_per_hidden_layer[1])))  # FIXME: Check if this is correct




if __name__ == "__main__":
    test_data = pd.read_csv("IRIS.csv")

    faouzia = Faouzia(test_data, 4)
    faouzia.execute_lifecycle()