import numpy as np
import pandas as pd

from faouzia import Faouzia

from models.model_config import ModelConfig
from utils.logger import logger
from utils.initialize_parameters_utils import initialize_parameters_random_normal


class FaouziaTabularClassifier(Faouzia):
    """
    This child class of Faouzia is responsible for handling classification tasks for both binary and multi-class
    classification problems on tabular data.
    """

    def __init__(self, data: pd.DataFrame) -> None:
            """
            Constructs a new FaouziaTabularClassifier object.
            """

            logger.info('Initializing FaouziaTabularClassifier object...')
            
            super().__init__(data)

    def preprocess_data(self) -> np.ndarray:
        """
        This method preprocesses the dataset according to user specifications.

        Preprocessing steps include:
            - Detect binary or multi-class classification problem and create labels map
        
        Returns:
            np.ndarray: Preprocessed dataset.
        """

        logger.info('Preprocessing data...')

        data_copy = self.data.copy()

        logger.info('Detecting binary or multi-class classification problem...')

        # NOTE: assume the last column is the labels column for now (this will be changed later)
        if data_copy.iloc[:, -1].nunique() == 2:
            logger.info('Detected binary classification problem.')

            self.labels_map = {data_copy.iloc[:, -1].unique()[0]: 0, data_copy.iloc[:, -1].unique()[1]: 1}

        else:
            logger.info('Detected multi-class classification problem.')

            # NOTE: may want to make the keys one-hot encoded vectors instead of integers
            self.labels_map = {data_copy.iloc[:, -1].unique()[i]: i for i in range(data_copy.iloc[:, -1].nunique())}

        logger.debug(f'labels map: {self.labels_map}')
        logger.info('Converting labels column using labels map...')

        data_copy.iloc[:, -1] = data_copy.iloc[:, -1].map(self.labels_map)

        # TODO: extract features from dataset and store in attribute (storing in attribute is hacky)

        return data_copy.values
    
    def model_selection(self, data_preprocessed: np.ndarray) -> ModelConfig:
        """
        This method selects the model configuration and architecture using preprocessed data

        Parameters:
            data_preprocessed: Preprocessed data

        Returns:
            ModelConfig: Configuration of the model architecture including hyperparameters
        """

        # TODO: Add more complex architecture selection decisioning logic (random search or optimization algorithm that
        #  selects an optimal model configuration)

        model_config = ModelConfig(
            num_nodes_per_layer={1: data_preprocessed.shape[1], 2: (np.unique(val[:, -1]).size())},
            activation_function_per_layer={1: 'relu', 2: 'softmax'},
            learning_rate=0.001,
            num_epochs=1000,
            batch_size=32,
            optimizer='adam',
            loss_function='categorical_cross_entropy',
            init_method='random_normal',
            regularization_method='l2'
        )

        return model_config

    # TODO: add features, weights and bias to method signature here and in base class
    def train_model(self, model_config: ModelConfig) -> np.ndarray:
        """
        This method trains the model (updates weights and bias)

        Parameters:
            model_config: Configuration of the model architecture including hyperparameters

        Returns:
            np.ndarray:
        """

        # TODO: Use eval function to select the init method in the model_config instead of explicitly selecting
        # init method here
        weights, bias = initialize_parameters_random_normal(model_config)

        # TODO: add inputs
        predicted_labels = self.forward_pass()

        # TODO: add inputs
        accuracy = self.calculate_loss()

        # backprop
        weights_updated, bias_updated = self.backpropagation()

        epoch_ctr = 0

        while epoch_ctr < model_config.num_epochs:

            predicted_labels = self.forward_pass()

            accuracy = self.calculate_loss()

            weights_updated, bias_updated = self.backpropagation()

            epoch_ctr += 1

        return

    # TODO: Implement this method
    def evaluate_model(self, model_config: ModelConfig, weights: np.ndarray, bias: np.ndarray) -> float:

        return None
