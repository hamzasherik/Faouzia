
import numpy as np
import pandas as pd

from faouzia import Faouzia
from models.model_config import ModelConfig
from utils.logger import logger


class FaouziaTabularClassifier(Faouzia):
    """
    This child class of Faouzia is responsible for handling classification tasks for both binary and multi-class classification problems on tabular data.
    """

    def __init__(self, data: pd.DataFrame) -> None:
            """
            Constructs a new FaouziaTabularClassifier object.
            """
            
            super().__init__(data)

            logger.info("Initializing FaouziaTabularClassifier object...")
            
            return

    # TODO: implement all preprocessing steps for tabular data
    def preprocess_data(self) -> np.ndarray:
        """
        This method preprocesses the dataset according to user specifications.

        Preprocessing steps include:
            - Detecting binary or multi-class classification problem and creating labels map
            - 
            -
        
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

        logger.info('Converting labels column using labels map...')

        data_copy.iloc[:, -1] = data_copy.iloc[:, -1].map(self.labels_map)

        return
    
    def model_selection(self) -> ModelConfig:
        
        return
    
    def train_model(self, model_config: ModelConfig, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        
        return
    
    def evaluate_model(self, model_config: ModelConfig, weights: np.ndarray, bias: np.ndarray) -> float:

        return
    