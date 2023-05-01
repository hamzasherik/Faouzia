import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from utils.logger import logger
from models.model_config import ModelConfig

# NOTE: Start off by implementing classification and regression for tabular data
# TODO: Complete docstring
class Faouzia(ABC):
    """
    This is the Faouzia base class which will support concrete implementations for supervised and unsuperivsed learning and
    all data types (tabular, image, text, etc.).
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Constructs a new Faouzia object.

        Parameters:
            data (pd.DataFrame): Dataframe containing the dataset.
        """
        
        logger.info("Initializing Faouzia object...")

        self.data = data
 
    # NOTE: preprocess data in pandas then convert to numpy for model training
    @abstractmethod
    def preprocess_data(self) -> None:
        """
        This method preprocesses the dataset according to user specifications.

        The preprocessing steps will vary depending on the concrete implementation and the data.
        """

        pass

    @abstractmethod
    def model_selection(self) -> ModelConfig:
        """
        This method selects the model to be used for training and evaluation.

        This method includes hyperparameter selection, architecture selection, hyperparameter tuning, etc. The specific steps
        will vary depending on the concrete implementation and the data.

        Returns:
            ModelConfig: Model configuration containing the hyperparameters, architecture, etc. of the deep learning model.
        """

        pass

    @abstractmethod
    def train_model(self, model_config: ModelConfig, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
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

        pass

    @abstractmethod
    def evaluate_model(self, model_config: ModelConfig, weights: np.ndarray, bias: np.ndarray) -> float:
        """
        This method evaluates a deep learning model using the given hyperparameters, weights and bias.

        Parameters:
            hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.
            weights (np.ndarray): Weights of the deep learning model.
            bias (np.ndarray): Bias of the deep learning model.

        Returns:
            float: Evaluation of the deep learning model.
        """

        pass

    # NOTE: May want to change evaluate model so that its where hyperparameter tuning is done and then create a predict method
