import pydantic_numpy.dtype as pnd

from pydantic import BaseModel, validator
from models.Hyperparameters import Hyperparameters


class Configuration(BaseModel):
    """
    This class is used to store the configuration of a deep learning model for Faouzia (including hyperparameters, parameters and 
    accuracy). It also includes both built-in.

    Attributes:
        hyperparameters (Hyperparameters): Hyperparameters of the deep learning model.
        weights (np.ndarray): Weights of the deep learning model.
        biases (np.ndarray): Biases of the deep learning model.
        accuracy (float): Accuracy of the deep learning model.
    """

    hyperparameters: Hyperparameters
    weights: pnd.NDArray
    biases: pnd.NDArray
    accuracy: float = 0.0

    @validator('accuracy', always=True)
    def accuracy_must_be_between_0_and_1(cls, accuracy: float) -> float:
        """
        This method validates that the accuracy is between 0 and 1.
        
        Parameters:
            accuracy (float): Accuracy of the deep learning model.
        
        Returns:
            float: Accuracy of the deep learning model.
        """

        if accuracy < 0 or accuracy > 1:
            raise ValueError('Accuracy must be between 0 and 1.')
        return accuracy

    # TODO: Validation for both weights and bias confirming dimensions are correct
