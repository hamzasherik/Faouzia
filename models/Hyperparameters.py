from pydantic import BaseModel, validator
from typing import Dict


class Hyperparameters(BaseModel):
    """
    This class is used to store the hyperparameters of a deep learning model for Faouzia. It also includes both built-in and custom data validations.

    Attributes:
        num_input_dimensions (int): Number of input dimensions.
        num_output_nodes (int): Number of output nodes.
        num_hidden_layers (int): Number of hidden layers.
        num_nodes_per_hidden_layer (dict[int, int]): Number of nodes per hidden layer.
        activation_function_per_layer (dict[int, str]): Activation function per layer.
        learning_rate (float): Learning rate.
        optimizer (str): Optimizer.
        loss_function (str): Loss function.
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        init_method (str): Initialization method.
        regularization_method (str): Regularization method.
    """

    num_input_dimensions: int
    num_output_nodes: int
    num_hidden_layers: int
    num_nodes_per_hidden_layer: Dict[int, int] 
    activation_function_per_layer: Dict[int, str]
    learning_rate: float
    optimizer: str
    loss_function: str
    num_epochs: int
    batch_size: int
    init_method: str
    regularization_method: str

    @validator('num_input_dimensions')
    def num_input_dimensions_must_be_greater_than_zero(cls, num_input_dimensions: int) -> int:
        """
        This method validates that num_input_dimensions is greater than zero.

        Parameters:
            num_input_dimensions (int): Number of input dimensions.

        Returns:
            int: Number of input dimensions.
        """

        if num_input_dimensions <= 0:
            raise ValueError('num_input_dimensions must be greater than zero')
        
        return num_input_dimensions
    
    @validator('num_output_nodes')
    def num_output_nodes_must_be_greater_than_zero(cls, num_output_nodes: int) -> int:
        """
        This method validates that num_output_nodes is greater than zero.
        
        Parameters:
            num_output_nodes (int): Number of output nodes.

        Returns:
            int: Number of output nodes.
        """

        if num_output_nodes <= 0:
            raise ValueError('num_output_nodes must be greater than zero')
        
        return num_output_nodes
    
    @validator('num_hidden_layers')
    def num_hidden_layers_must_be_greater_than_zero(cls, num_hidden_layers: int) -> int:
        """
        This method validates that num_hidden_layers is greater than zero.

        Parameters:
            num_hidden_layers (int): Number of hidden layers.

        Returns:
            int: Number of hidden layers.
        """

        if num_hidden_layers <= 0:
            raise ValueError('num_hidden_layers must be greater than zero')
        
        return num_hidden_layers
    
    @validator('num_nodes_per_hidden_layer')
    def num_nodes_per_hidden_layer_must_be_greater_than_zero(cls, num_nodes_per_hidden_layer: Dict[int, int]) -> Dict[int, int]:
        """
        This method validates that num_nodes_per_hidden_layer is greater than zero.

        Parameters:
            num_nodes_per_hidden_layer (dict[int, int]): Number of nodes per hidden layer.
        
        Returns:
            dict[int, int]: Number of nodes per hidden layer.
        """

        for value in num_nodes_per_hidden_layer.values():
            if value <= 0:
                raise ValueError('num_nodes_per_hidden_layer must be greater than zero')
            
        return num_nodes_per_hidden_layer
    
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
        valid_activation_functions = ['relu', 'sigmoid', 'tanh']
        
        for key, value in activation_function_per_layer.items():
            if value.casefold() not in valid_activation_functions:
                raise ValueError('activation_function_per_layer must be either relu, sigmoid, or tanh')
            
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
        valid_loss_functions = ['mse', 'binary_crossentropy', 'categorical_crossentropy']

        if loss_function.casefold() not in valid_loss_functions:
            raise ValueError('loss_function must be either mse or binary_crossentropy')
        
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