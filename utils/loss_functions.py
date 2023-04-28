import numpy as np

from utils.logger import logger

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This method calculates the categorical cross entropy loss.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Categorical cross entropy loss.
    """

    logger.info("Calculating categorical cross entropy loss...")

    return -np.sum(y_true * np.log(y_pred))

def categorical_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This method calculates the derivative of the categorical cross entropy loss.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Derivative of the categorical cross entropy loss.
    """

    logger.info("Calculating derivative of categorical cross entropy loss...")
    
    return y_pred - y_true