import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    """
    This method implements the ReLU activation function.

    Parameters:
        z (np.ndarray): Input of the activation function.

    Returns:
        np.ndarray: Output of the activation function.
    """

    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    This method implements the derivative of the ReLU activation function.

    Parameters:
        z (np.ndarray): Input of the activation function.

    Returns:
        np.ndarray: Output of the activation function.
    """

    return np.where(z > 0, 1, 0)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    This method implements the softmax activation function.

    Parameters:
        z (np.ndarray): Input of the activation function.

    Returns:
        np.ndarray: Output of the activation function.
    """

    z = np.array(z, dtype=float)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))

    return exp / np.sum(exp, axis=1, keepdims=True)


def softmax_derivative(z: np.ndarray) -> np.ndarray:
    """
    This method implements the derivative of the softmax activation function.

    Parameters:
        z (np.ndarray): Input of the activation function.

    Returns:
        np.ndarray: Output of the activation function.
    """

    return softmax(z) * (1 - softmax(z))