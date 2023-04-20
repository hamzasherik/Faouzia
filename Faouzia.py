import numpy as np
import pandas as pd
import pydantic

from typing import Optional
from abc import ABC
from utils.logger import logger
from utils.data_utils import preprocess_labels, labels_to_one_hot_map


class Faouzia(ABC):
    """
    This class is used to determine the optimal configuration of parameters and hyperparameters for a multi-class classification problem for tabular data only.

    This class does not include data preprocessing (e.g., dimensionality reduction, imputation, etc.). It is assumed that the data is already preprocessed and ready before passing to Faouzia.

    Attributes:
        data (np.ndarray): Tabular data stored in numpy array.
        labels_col_idx (int): Index of the labels column.
    """

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

        # TODO: Use both raw_data and data_copy or create data_copt in execute_lifecycle method

        self.raw_data = data.values
        self.data_copy = np.copy(self.raw_data)
        self.labels_column_index = labels_column_index

        self.hyperparameters = {}
        self.weights = []
        self.bias = []


    def execute_lifecycle(self) -> bool:
        """
        This method executes the lifecycle that determines the optimal configuration of parameters and hyperparameters for a deep learning model in the following steps:
            - Data preprocessing
            -

        Returns:
            bool: True if lifecycle was executed successfully, False otherwise.
        """

        logger.info("Executing lifecycle...")

        # TODO: Create one-hot labels map in constructor as an attribute?

        labels_map = labels_to_one_hot_map(self.data_copy, self.labels_column_index)

        self.initialize_hyperparameters()

        
    def initialize_hyperparameters(self) -> None:
        """
        This method initializes the hyperparameters of the deep learning model and applies those values to the object's hyperparameters attribute.
        """

        logger.info("Initializing hyperparameters...")

    
    def initialize_parameters(self) -> None:
        """
        This method initializes the parameters of the deep learning model and applies those values to the object's weights and bias attributes.
        """

        logger.info("Initializing parameters...")

    


if __name__ == "__main__":
    test_data = pd.read_csv("IRIS.csv")

    faouzia = Faouzia(test_data, 4)
    faouzia.execute_lifecycle()