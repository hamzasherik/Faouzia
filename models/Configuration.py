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
    loss: float

    @validator('loss', always=True)
    def accuracy_must_be_between_0_and_1(cls, loss: float) -> float:
        """
        This method validates that the loss is greater than or equal to 0
        
        Parameters:
            loss (float): Loss of the deep learning model.
        
        Returns:
            float: Loss of the deep learning model.
        """

        if loss < 0:
            raise ValueError('Accuracy must be between 0 or greater.')
        
        return loss

    # TODO: Validation for both weights and bias confirming dimensions are correct. Might need to use root_validator
