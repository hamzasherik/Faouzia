from pydantic import BaseModel, validator
from typing import Dict


# TODO: Convert to modelConfig
class Hyperparameters(BaseModel):
    """
    This class is used to store the hyperparameters of a deep learning model for Faouzia. It also includes both built-in and custom data validations.

    Attributes:
        num_nodes_per_layer (dict[int, int]): Number of nodes per layer.
        activation_function_per_layer (dict[int, str]): Activation function per layer.
        learning_rate (float): Learning rate.
        optimizer (str): Optimizer.
        loss_function (str): Loss function.
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        init_method (str): Initialization method.
        regularization_method (str): Regularization method.
    """

    num_nodes_per_layer: Dict[int, int]
    activation_function_per_layer: Dict[int, str]
    learning_rate: float
    optimizer: str
    loss_function: str
    num_epochs: int
    batch_size: int
    init_method: str
    regularization_method: str

    @validator('num_nodes_per_layer')
    def num_nodes_per_layer_must_be_greater_than_zero(cls, num_nodes_per_layer: Dict[int, int]) -> Dict[int, int]:
        """
        This method validates that num_nodes_per_layer is greater than zero.
        
        Parameters:
            num_nodes_per_layer (dict[int, int]): Number of nodes per layer.

        Returns:
            dict[int, int]: Number of nodes per layer.
        """

        for key, value in num_nodes_per_layer.items():
            if value <= 0:
                raise ValueError(f'{value} is an invalid number of nodes for layer {key}')
            
        return num_nodes_per_layer
    
    @validator('activation_function_per_layer')
    def activation_function_per_layer_must_be_valid(cls, activation_function_per_layer: Dict[int, str]) -> Dict[int, str]:
        """
        This method validates that activation_function_per_layer is either relu, sigmoid, or tanh.
        
        Parameters:
            activation_function_per_layer (dict[int, str]): Activation function per layer.

        Returns:
            dict[int, str]: Activation function per layer.
        """
        # TODO: Move valid methods to constants file and update docstring.
        valid_activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
        
        for key, value in activation_function_per_layer.items():
            if value.casefold() not in valid_activation_functions:
                raise ValueError(f'{value} is an invalid activation function for layer {key}')
            
            else:
                activation_function_per_layer[key] = value.casefold()
            
        return activation_function_per_layer
    
    @validator('learning_rate')
    def learning_rate_must_be_greater_than_zero(cls, learning_rate: float) -> float:
        """
        This method validates that learning_rate is greater than zero.
        
        Parameters:
            learning_rate (float): Learning rate.

        Returns:
            float: Learning rate.
        """

        if learning_rate <= 0:
            raise ValueError('learning_rate must be greater than zero')
        
        return learning_rate

    @validator('optimizer')
    def optimizer_must_be_valid(cls, optimizer: str) -> str:
        """
        This method validates that optimizer is either adam or sgd.

        Parameters:
            optimizer (str): Optimizer.

        Returns:
            str: Optimizer.
        """
        # TODO: Move valid methods to constants file and update docstring.
        valid_optimizers = ['adam', 'sgd']

        if optimizer.casefold() not in valid_optimizers:
            raise ValueError('optimizer must be either adam or sgd')
        
        return optimizer.casefold()
    
    @validator('loss_function')
    def loss_function_must_be_valid(cls, loss_function: str) -> str:
        """
        This method validates that loss_function is either mse or binary_crossentropy.
        
        Parameters:
            loss_function (str): Loss function.

        Returns:
            str: Loss function.
        """
        # TODO: Move valid methods to constants file and update docstring.
        valid_loss_functions = ['mse', 'categorical_cross_entropy']

        if loss_function.casefold() not in valid_loss_functions:
            raise ValueError(f'{loss_function} is an invalid loss function')
        
        return loss_function.casefold()
    
    @validator('num_epochs')
    def num_epochs_must_be_greater_than_zero(cls, num_epochs: int) -> int:
        """
        This method validates that num_epochs is greater than zero.

        Parameters:
            num_epochs (int): Number of epochs.

        Returns:   
            int: Number of epochs.
        """

        if num_epochs <= 0:
            raise ValueError('num_epochs must be greater than zero')
        
        return num_epochs
    
    @validator('batch_size')
    def batch_size_must_be_greater_than_zero(cls, batch_size: int) -> int:
        """
        This method validates that batch_size is greater than zero.
        
        Parameters:
            batch_size (int): Batch size.
            
        Returns:
            int: Batch size.
        """

        if batch_size <= 0:
            raise ValueError('batch_size must be greater than zero')
        
        return batch_size
    
    @validator('init_method')
    def init_method_must_be_valid(cls, init_method: str) -> str:
        """
        This method validates that init_method is either he_uniform or glorot_uniform.
        
        Parameters:
            init_method (str): Initialization method.

        Returns:
            str: Initialization method.
        """
        # TODO: Move valid methods to constants file and update docstring.
        valid_init_methods = ['he_uniform', 'glorot_uniform', 'random_normal']

        if init_method.casefold() not in valid_init_methods:
            raise ValueError('init_method must be either he_uniform or glorot_uniform')
        
        return init_method.casefold()
    
    @validator('regularization_method')
    def regularization_method_must_be_valid(cls, regularization_method: str) -> str:
        """
        This method validates that regularization_method is either l1, l2, or l1_l2.
        
        Parameters:
            regularization_method (str): Regularization method.

        Returns:
            str: Regularization method.
        """
        # TODO: Move valid methods to constants file and update docstring.
        valid_regularization_methods = ['l1', 'l2', 'l1_l2']

        if regularization_method.casefold() not in valid_regularization_methods:
            raise ValueError('regularization_method must be either l1, l2, or l1_l2')
        
        return regularization_method.casefold()
    
    # TODO: Validation to confirm num_nodes_per_layer is consistent with activation_function_per_layer.