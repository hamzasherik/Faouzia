import numpy as np

from models.model_config import ModelConfig
from utils.logger import logger


def initialize_parameters_random_normal(model_config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    This method initializes the parameters (weights and bias) of the deep learning model using the random normal method.

    Parameters:
        model_config: Configuration of the model architecture including hyperparameters

    Returns:
        np.ndarray: Weights of the deep learning model.
        np.ndarray: Bias of the deep learning model.
    """

    logger.info("Initializing parameters using random normal method...")

    weights = []
    bias = []

    for key, value in model_config.num_nodes_per_layer.items():

        if key == len(model_config.num_nodes_per_layer):
            continue

        weights.append(np.random.randn(value, model_config.num_nodes_per_layer[key + 1]))
        bias.append(np.full((1, model_config.num_nodes_per_layer[key + 1]), 0.1))

    # TODO: confirm type returned matches type hint in signature
    return weights, bias
