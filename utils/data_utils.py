import numpy as np

from utils.logger import logger
from typing import Dict


# TODO: Update docstring
def labels_to_one_hot_map(data: np.ndarray, labels_column_index: int) -> Dict[str, np.ndarray]:
        """
        This method creates a dictionary that maps labels to one-hot vectors.

        Parameters:
            data (np.ndarray): Tabular data stored in numpy array.
            labels_column_index (int): Index of the labels column.
        
        Returns:
            dict: Dictionary of one-hot vectors.
        """

        logger.info("Creating dictionary that maps labels to one-hot vectors...")

        labels = data[:, labels_column_index]
        unique_labels = np.unique(labels)
     
        one_hot_dict = {}

        for label in unique_labels:
            one_hot = np.zeros(len(unique_labels), dtype=int)
            one_hot[np.where(unique_labels == label)] = 1
            one_hot_dict[label] = one_hot

        logger.debug(f'Labels to one-hot map: {one_hot_dict}')

        return one_hot_dict


# TODO: Update docstring
def preprocess_labels(data: np.ndarray, labels_column_index: int, labels_to_one_hot_map: Dict[str, np.ndarray]) -> np.ndarray:
        """
        This method preprocesses raw labels (categorical or numerical) by converting them to one-hot vectors.

        Parameters:
            data (np.ndarray): Tabular data stored in numpy array.
            labels_column_index (int): Index of the labels column.
        
        Returns:
            np.ndarray: Preprocessed labels stored in numpy array.
        """

        logger.info("Preprocessing data by one-hot encoding labels...")

        labels = data[:, labels_column_index]

        one_hot_labels = np.array([labels_to_one_hot_map[label] for label in labels])

        return one_hot_labels