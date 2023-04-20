from utils.logger import logger
import numpy as np

def preprocess_labels(data: np.ndarray, labels_column_index: int) -> np.ndarray:
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
        unique_labels = np.unique(labels)
     
        data_one_hot_labels = np.zeros((labels.shape[0], unique_labels.shape[0]))

        for i in range(labels.shape[0]):
            data_one_hot_labels[i, labels[i]] = 1

        return data_one_hot_labels


def labels_to_one_hot_map(data: np.ndarray, labels_column_index: int) -> dict:
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