import numpy as np

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This method calculates the categorical cross entropy loss.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Categorical cross entropy loss.
    """

    # TODO: calculate categorical cross entropy loss for each node