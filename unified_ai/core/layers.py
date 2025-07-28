"""
Neural network layers for the Unified AI Framework.
Provides various layer types including Dense, Convolutional, Normalization, and Attention layers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Callable, Any
from .tensor import Tensor, randn, zeros, ones
import math


class Layer(ABC):
    """
    Abstract base class for all neural network layers.
    
    All layers should inherit from this class and implement the forward and backward methods.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the layer.
        
        Args:
            name: Optional name for the layer
        """
        self.name = name or self.__class__.__name__
        self.trainable = True
        self.built = False
        self.input_shape = None
        self.output_shape = None
        
    @abstractmethod
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            training: Whether the layer is in training mode
            
        Returns:
            Output tensor
        """
        pass
    
    def build(self, input_shape: Tuple[int, ...]):
        """
        Build the layer (initialize weights and biases).
        
        Args:
            input_shape: Shape of the input tensor
        """
        self.input_shape = input_shape
        self.built = True
    
    def get_weights(self) -> list:
        """Get all trainable weights."""
        return []
    
    def set_weights(self, weights: list):
        """Set all trainable weights."""
        pass
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        return {'name': self.name}
    
    def __call__(self, x: Tensor, training: bool = True) -> Tensor:
        """Make the layer callable."""
        if not self.built:
            self.build(x.shape)
        return self.forward(x, training)


class Dense(Layer):
    """
    Fully connected (dense) layer.
    
    Performs the operation: output = activation(dot(input, kernel) + bias)
    """
    
    def __init__(self, units: int, activation: Optional[Union[str, Callable]] = None,
                 use_bias: bool = True, kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros', name: Optional[str] = None):
        """
        Initialize Dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'softmax', etc.)
            use_bias: Whether to use bias
            kernel_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            name: Layer name
        """
        super().__init__(name)
        self.units = units
        self.activation = self._get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        self.kernel = None
        self.bias = None
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer weights."""
        super().build(input_shape)
        
        input_dim = input_shape[-1]
        
        # Initialize kernel (weights)
        if self.kernel_initializer == 'glorot_uniform':
            limit = math.sqrt(6.0 / (input_dim + self.units))
            self.kernel = Tensor(
                np.random.uniform(-limit, limit, (input_dim, self.units)),
                requires_grad=True
            )
        elif self.kernel_initializer == 'he_normal':
            std = math.sqrt(2.0 / input_dim)
            self.kernel = Tensor(
                np.random.normal(0, std, (input_dim, self.units)),
                requires_grad=True
            )
        else:  # Default to normal
            self.kernel = randn(input_dim, self.units, requires_grad=True)
        
        # Initialize bias
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                self.bias = zeros(self.units, requires_grad=True)
            else:
                self.bias = randn(self.units, requires_grad=True)
        
        self.output_shape = input_shape[:-1] + (self.units,)
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Forward pass."""
        output = x @ self.kernel
        
        if self.use_bias:
            output = output + self.bias
        
        if self.activation:
            output = self.activation(output)
        
        return output
    
    def get_weights(self) -> list:
        """Get trainable weights."""
        weights = [self.kernel]
        if self.use_bias:
            weights.append(self.bias)
        return weights
    
    def set_weights(self, weights: list):
        """Set trainable weights."""
        self.kernel = weights[0]
        if self.use_bias and len(weights) > 1:
            self.bias = weights[1]
    
    def _get_activation(self, activation):
        """Get activation function."""
        if activation is None:
            return None
        elif isinstance(activation, str):
            return get_activation(activation)
        else:
            return activation


class Conv2D(Layer):
    """
    2D Convolutional layer.
    
    Applies 2D convolution over input images.
    """
    
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]] = 1,
                 padding: str = 'valid', activation: Optional[Union[str, Callable]] = None,
                 use_bias: bool = True, name: Optional[str] = None):
        """
        Initialize Conv2D layer.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of the convolution kernel
            strides: Stride of the convolution
            padding: Padding mode ('valid' or 'same')
            activation: Activation function
            use_bias: Whether to use bias
            name: Layer name
        """
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.lower()
        self.activation = self._get_activation(activation)
        self.use_bias = use_bias
        
        self.kernel = None
        self.bias = None
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer weights."""
        super().build(input_shape)
        
        # Assume input shape is (batch, height, width, channels)
        if len(input_shape) != 4:
            raise ValueError(f"Conv2D expects 4D input, got {len(input_shape)}D")
        
        _, input_height, input_width, input_channels = input_shape
        
        # Initialize kernel
        kernel_shape = (*self.kernel_size, input_channels, self.filters)
        fan_in = np.prod(kernel_shape[:-1])
        fan_out = np.prod(kernel_shape[:-2]) * self.filters
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        
        self.kernel = Tensor(
            np.random.uniform(-limit, limit, kernel_shape),
            requires_grad=True
        )
        
        # Initialize bias
        if self.use_bias:
            self.bias = zeros(self.filters, requires_grad=True)
        
        # Calculate output shape
        if self.padding == 'same':
            output_height = math.ceil(input_height / self.strides[0])
            output_width = math.ceil(input_width / self.strides[1])
        else:  # 'valid'
            output_height = math.ceil((input_height - self.kernel_size[0] + 1) / self.strides[0])
            output_width = math.ceil((input_width - self.kernel_size[1] + 1) / self.strides[1])
        
        self.output_shape = (input_shape[0], output_height, output_width, self.filters)
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Forward pass using simple convolution implementation."""
        batch_size, input_height, input_width, input_channels = x.shape
        kernel_height, kernel_width, _, _ = self.kernel.shape
        
        # Calculate padding
        if self.padding == 'same':
            pad_h = max(0, (self.output_shape[1] - 1) * self.strides[0] + kernel_height - input_height)
            pad_w = max(0, (self.output_shape[2] - 1) * self.strides[1] + kernel_width - input_width)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # Pad input
            x_padded = Tensor(np.pad(x.data, 
                                   ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                   mode='constant'), 
                            requires_grad=x.requires_grad)
        else:
            x_padded = x
        
        output_height, output_width = self.output_shape[1], self.output_shape[2]
        output = zeros(batch_size, output_height, output_width, self.filters, requires_grad=True)
        
        # Perform convolution
        for b in range(batch_size):
            for f in range(self.filters):
                for oh in range(output_height):
                    for ow in range(output_width):
                        h_start = oh * self.strides[0]
                        h_end = h_start + kernel_height
                        w_start = ow * self.strides[1]
                        w_end = w_start + kernel_width
                        
                        if h_end <= x_padded.shape[1] and w_end <= x_padded.shape[2]:
                            patch = x_padded[b, h_start:h_end, w_start:w_end, :]
                            kernel_f = self.kernel[:, :, :, f]
                            conv_result = (patch * kernel_f).sum()
                            output.data[b, oh, ow, f] = conv_result.data
        
        if self.use_bias:
            output = output + self.bias
        
        if self.activation:
            output = self.activation(output)
        
        return output
    
    def get_weights(self) -> list:
        """Get trainable weights."""
        weights = [self.kernel]
        if self.use_bias:
            weights.append(self.bias)
        return weights
    
    def _get_activation(self, activation):
        """Get activation function."""
        if activation is None:
            return None
        elif isinstance(activation, str):
            return get_activation(activation)
        else:
            return activation


class BatchNorm(Layer):
    """
    Batch Normalization layer.
    
    Normalizes inputs by maintaining running statistics of mean and variance.
    """
    
    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-3,
                 center: bool = True, scale: bool = True, name: Optional[str] = None):
        """
        Initialize BatchNorm layer.
        
        Args:
            momentum: Momentum for moving average
            epsilon: Small constant for numerical stability
            center: Whether to use beta parameter
            scale: Whether to use gamma parameter
            name: Layer name
        """
        super().__init__(name)
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_var = None
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer parameters."""
        super().build(input_shape)
        
        # Parameters shape is the last dimension
        param_shape = (input_shape[-1],)
        
        if self.scale:
            self.gamma = ones(*param_shape, requires_grad=True)
        
        if self.center:
            self.beta = zeros(*param_shape, requires_grad=True)
        
        # Moving statistics (not trainable)
        self.moving_mean = zeros(*param_shape)
        self.moving_var = ones(*param_shape)
        
        self.output_shape = input_shape
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Forward pass."""
        if training:
            # Calculate batch statistics
            axes = tuple(range(len(x.shape) - 1))  # All axes except the last one
            batch_mean = x.mean(axis=axes, keepdims=True)
            batch_var = ((x - batch_mean) ** 2).mean(axis=axes, keepdims=True)
            
            # Update moving statistics
            self.moving_mean.data = (self.momentum * self.moving_mean.data + 
                                   (1 - self.momentum) * batch_mean.data.squeeze())
            self.moving_var.data = (self.momentum * self.moving_var.data + 
                                  (1 - self.momentum) * batch_var.data.squeeze())
            
            # Normalize using batch statistics
            x_norm = (x - batch_mean) / Tensor(np.sqrt(batch_var.data + self.epsilon))
        else:
            # Use moving statistics for inference
            mean = self.moving_mean.reshape((1,) * (len(x.shape) - 1) + (-1,))
            var = self.moving_var.reshape((1,) * (len(x.shape) - 1) + (-1,))
            x_norm = (x - mean) / Tensor(np.sqrt(var.data + self.epsilon))
        
        # Scale and shift
        output = x_norm
        if self.scale:
            gamma = self.gamma.reshape((1,) * (len(x.shape) - 1) + (-1,))
            output = output * gamma
        
        if self.center:
            beta = self.beta.reshape((1,) * (len(x.shape) - 1) + (-1,))
            output = output + beta
        
        return output
    
    def get_weights(self) -> list:
        """Get trainable weights."""
        weights = []
        if self.scale:
            weights.append(self.gamma)
        if self.center:
            weights.append(self.beta)
        return weights


class Dropout(Layer):
    """
    Dropout layer for regularization.
    
    Randomly sets input units to 0 with a frequency of rate at each step during training.
    """
    
    def __init__(self, rate: float, name: Optional[str] = None):
        """
        Initialize Dropout layer.
        
        Args:
            rate: Dropout rate (fraction of input units to drop)
            name: Layer name
        """
        super().__init__(name)
        self.rate = rate
        
        if not 0 <= rate <= 1:
            raise ValueError(f"Dropout rate must be between 0 and 1, got {rate}")
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer."""
        super().build(input_shape)
        self.output_shape = input_shape
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Forward pass."""
        if not training or self.rate == 0:
            return x
        
        # Generate dropout mask
        keep_prob = 1 - self.rate
        mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
        
        return x * Tensor(mask)


class Embedding(Layer):
    """
    Embedding layer for converting integer indices to dense vectors.
    
    Turns positive integers (indexes) into dense vectors of fixed size.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 embeddings_initializer: str = 'uniform', name: Optional[str] = None):
        """
        Initialize Embedding layer.
        
        Args:
            input_dim: Size of the vocabulary
            output_dim: Size of the dense vector
            embeddings_initializer: Initializer for embeddings
            name: Layer name
        """
        super().__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        
        self.embeddings = None
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer."""
        super().build(input_shape)
        
        # Initialize embeddings
        if self.embeddings_initializer == 'uniform':
            self.embeddings = Tensor(
                np.random.uniform(-0.05, 0.05, (self.input_dim, self.output_dim)),
                requires_grad=True
            )
        else:  # Default to normal
            self.embeddings = randn(self.input_dim, self.output_dim, requires_grad=True)
        
        self.output_shape = input_shape + (self.output_dim,)
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Forward pass."""
        # Convert indices to embeddings
        indices = x.data.astype(int)
        output_data = self.embeddings.data[indices]
        
        output = Tensor(output_data, requires_grad=self.embeddings.requires_grad)
        
        # Set up gradient computation
        if self.embeddings.requires_grad:
            def grad_fn(grad):
                embed_grad = np.zeros_like(self.embeddings.data)
                np.add.at(embed_grad, indices, grad.data)
                self.embeddings.backward(Tensor(embed_grad))
            
            output._grad_fn = grad_fn
            output._children = [self.embeddings]
        
        return output
    
    def get_weights(self) -> list:
        """Get trainable weights."""
        return [self.embeddings]


class MultiHeadAttention(Layer):
    """
    Multi-Head Attention layer.
    
    Implements the multi-head attention mechanism from "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 name: Optional[str] = None):
        """
        Initialize MultiHeadAttention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            name: Layer name
        """
        super().__init__(name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_k = d_model // num_heads
        
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.dropout = Dropout(dropout) if dropout > 0 else None
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer."""
        super().build(input_shape)
        
        # Initialize weight matrices
        limit = math.sqrt(6.0 / (2 * self.d_model))
        
        self.W_q = Tensor(
            np.random.uniform(-limit, limit, (self.d_model, self.d_model)),
            requires_grad=True
        )
        self.W_k = Tensor(
            np.random.uniform(-limit, limit, (self.d_model, self.d_model)),
            requires_grad=True
        )
        self.W_v = Tensor(
            np.random.uniform(-limit, limit, (self.d_model, self.d_model)),
            requires_grad=True
        )
        self.W_o = Tensor(
            np.random.uniform(-limit, limit, (self.d_model, self.d_model)),
            requires_grad=True
        )
        
        if self.dropout:
            self.dropout.build(input_shape)
        
        self.output_shape = input_shape
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, training: bool = True) -> Tensor:
        """Forward pass."""
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations
        Q = x @ self.W_q  # (batch_size, seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose((0, 2, 1, 3))
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose((0, 2, 1, 3))
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose((0, 2, 1, 3))
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask, training)
        
        # Concatenate heads
        attention_output = attention_output.transpose((0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear transformation
        output = attention_output @ self.W_o
        
        return output
    
    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, 
                                    mask: Optional[Tensor] = None, training: bool = True) -> Tensor:
        """Compute scaled dot-product attention."""
        # Compute attention scores
        scores = Q @ K.transpose((0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Apply softmax
        attention_weights = softmax(scores, axis=-1)
        
        # Apply dropout
        if self.dropout and training:
            attention_weights = self.dropout(attention_weights, training)
        
        # Apply attention to values
        output = attention_weights @ V
        
        return output
    
    def get_weights(self) -> list:
        """Get trainable weights."""
        return [self.W_q, self.W_k, self.W_v, self.W_o]


# Activation functions
def relu(x: Tensor) -> Tensor:
    """ReLU activation function."""
    result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad):
            x_grad = grad.data * (x.data > 0)
            x.backward(Tensor(x_grad))
        
        result._grad_fn = grad_fn
        result._children = [x]
    
    return result


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation function."""
    sigmoid_data = 1 / (1 + np.exp(-np.clip(x.data, -500, 500)))
    result = Tensor(sigmoid_data, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad):
            x_grad = grad.data * sigmoid_data * (1 - sigmoid_data)
            x.backward(Tensor(x_grad))
        
        result._grad_fn = grad_fn
        result._children = [x]
    
    return result


def tanh(x: Tensor) -> Tensor:
    """Tanh activation function."""
    tanh_data = np.tanh(x.data)
    result = Tensor(tanh_data, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad):
            x_grad = grad.data * (1 - tanh_data ** 2)
            x.backward(Tensor(x_grad))
        
        result._grad_fn = grad_fn
        result._children = [x]
    
    return result


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation function."""
    # Subtract max for numerical stability
    x_max = Tensor(np.max(x.data, axis=axis, keepdims=True))
    x_shifted = x - x_max
    
    exp_x = Tensor(np.exp(x_shifted.data))
    sum_exp = exp_x.sum(axis=axis, keepdims=True)
    
    result = exp_x / sum_exp
    return result


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Leaky ReLU activation function."""
    result_data = np.where(x.data > 0, x.data, alpha * x.data)
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad):
            x_grad = grad.data * np.where(x.data > 0, 1, alpha)
            x.backward(Tensor(x_grad))
        
        result._grad_fn = grad_fn
        result._children = [x]
    
    return result


def get_activation(name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'softmax': softmax,
        'leaky_relu': leaky_relu,
        'linear': lambda x: x,
        None: lambda x: x
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name]