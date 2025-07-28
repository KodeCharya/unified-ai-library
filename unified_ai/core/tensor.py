"""
Tensor abstraction for Unified AI Framework.
Provides a NumPy-based tensor with automatic differentiation support.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any
import warnings


class Tensor:
    """
    A tensor class that wraps NumPy arrays with automatic differentiation support.
    
    This class provides the foundation for all computations in the Unified AI framework,
    supporting both forward and backward passes for gradient computation.
    """
    
    def __init__(self, data: Union[np.ndarray, list, float, int], 
                 requires_grad: bool = False, dtype: Optional[np.dtype] = None):
        """
        Initialize a tensor.
        
        Args:
            data: The tensor data (numpy array, list, or scalar)
            requires_grad: Whether to compute gradients for this tensor
            dtype: Data type for the tensor
        """
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=dtype)
        elif isinstance(data, (int, float)):
            data = np.array(data, dtype=dtype)
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
            
        self.data = data.astype(dtype) if dtype else data
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._children = []
        
        if requires_grad:
            self.grad = np.zeros_like(self.data)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim
    
    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self.data.size
    
    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return self.data.dtype
    
    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array."""
        return self.data
    
    def item(self) -> Union[int, float]:
        """Return the tensor as a Python scalar (for single-element tensors)."""
        return self.data.item()
    
    def reshape(self, *shape) -> 'Tensor':
        """Reshape the tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def grad_fn(grad):
                return grad.reshape(self.shape)
            result._grad_fn = grad_fn
            result._children = [self]
            
        return result
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Transpose the tensor."""
        result = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def grad_fn(grad):
                if axes is None:
                    return np.transpose(grad)
                else:
                    # Reverse the permutation
                    reverse_axes = np.argsort(axes)
                    return np.transpose(grad, reverse_axes)
            result._grad_fn = grad_fn
            result._children = [self]
            
        return result
    
    @property
    def T(self) -> 'Tensor':
        """Transpose property (2D only)."""
        return self.transpose()
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> 'Tensor':
        """Sum along specified axes."""
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                       requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def grad_fn(grad):
                if axis is None:
                    return np.full_like(self.data, grad.data)
                else:
                    # Expand dimensions that were summed
                    grad_expanded = grad.data
                    if not keepdims:
                        if isinstance(axis, int):
                            grad_expanded = np.expand_dims(grad_expanded, axis)
                        else:
                            for ax in sorted(axis):
                                grad_expanded = np.expand_dims(grad_expanded, ax)
                    
                    return np.broadcast_to(grad_expanded, self.shape)
            
            result._grad_fn = grad_fn
            result._children = [self]
            
        return result
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
             keepdims: bool = False) -> 'Tensor':
        """Mean along specified axes."""
        result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), 
                       requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def grad_fn(grad):
                if axis is None:
                    n = self.data.size
                    return np.full_like(self.data, grad.data / n)
                else:
                    # Calculate the number of elements along the reduced axes
                    if isinstance(axis, int):
                        n = self.shape[axis]
                    else:
                        n = np.prod([self.shape[ax] for ax in axis])
                    
                    grad_expanded = grad.data / n
                    if not keepdims:
                        if isinstance(axis, int):
                            grad_expanded = np.expand_dims(grad_expanded, axis)
                        else:
                            for ax in sorted(axis):
                                grad_expanded = np.expand_dims(grad_expanded, ax)
                    
                    return np.broadcast_to(grad_expanded, self.shape)
            
            result._grad_fn = grad_fn
            result._children = [self]
            
        return result
    
    def backward(self, grad: Optional['Tensor'] = None):
        """
        Compute gradients using backpropagation.
        
        Args:
            grad: Gradient from the next layer (defaults to ones for scalar outputs)
        """
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.data.size == 1:
                grad = Tensor(np.ones_like(self.data))
            else:
                raise RuntimeError("grad must be specified for non-scalar tensors")
        
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        
        self.grad += grad.data
        
        # Propagate gradients to children
        if self._grad_fn is not None:
            for child in self._children:
                if child.requires_grad:
                    child_grad = self._grad_fn(grad)
                    child.backward(Tensor(child_grad))
    
    def zero_grad(self):
        """Reset gradients to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
    
    def detach(self) -> 'Tensor':
        """Return a new tensor detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def clone(self) -> 'Tensor':
        """Create a copy of the tensor."""
        result = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        return result
    
    # Arithmetic operations
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Addition operation."""
        if isinstance(other, (int, float)):
            other = Tensor(other)
        elif not isinstance(other, Tensor):
            return NotImplemented
        
        result = Tensor(self.data + other.data, 
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def grad_fn(grad):
                self_grad = grad.data
                other_grad = grad.data
                
                # Handle broadcasting
                if self.shape != grad.shape:
                    # Sum over broadcasted dimensions
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(self.shape) or self.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        self_grad = np.sum(self_grad, axis=tuple(axes_to_sum), keepdims=True)
                    self_grad = self_grad.reshape(self.shape)
                
                if other.shape != grad.shape:
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(other.shape) or other.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        other_grad = np.sum(other_grad, axis=tuple(axes_to_sum), keepdims=True)
                    other_grad = other_grad.reshape(other.shape)
                
                if self.requires_grad:
                    self.backward(Tensor(self_grad))
                if other.requires_grad:
                    other.backward(Tensor(other_grad))
            
            result._grad_fn = grad_fn
            result._children = [self, other]
        
        return result
    
    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        """Reverse addition."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Subtraction operation."""
        if isinstance(other, (int, float)):
            other = Tensor(other)
        elif not isinstance(other, Tensor):
            return NotImplemented
        
        result = Tensor(self.data - other.data, 
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def grad_fn(grad):
                self_grad = grad.data
                other_grad = -grad.data
                
                # Handle broadcasting (same as addition)
                if self.shape != grad.shape:
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(self.shape) or self.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        self_grad = np.sum(self_grad, axis=tuple(axes_to_sum), keepdims=True)
                    self_grad = self_grad.reshape(self.shape)
                
                if other.shape != grad.shape:
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(other.shape) or other.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        other_grad = np.sum(other_grad, axis=tuple(axes_to_sum), keepdims=True)
                    other_grad = other_grad.reshape(other.shape)
                
                if self.requires_grad:
                    self.backward(Tensor(self_grad))
                if other.requires_grad:
                    other.backward(Tensor(other_grad))
            
            result._grad_fn = grad_fn
            result._children = [self, other]
        
        return result
    
    def __rsub__(self, other: Union[float, int]) -> 'Tensor':
        """Reverse subtraction."""
        return Tensor(other).__sub__(self)
    
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise multiplication."""
        if isinstance(other, (int, float)):
            other = Tensor(other)
        elif not isinstance(other, Tensor):
            return NotImplemented
        
        result = Tensor(self.data * other.data, 
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def grad_fn(grad):
                self_grad = grad.data * other.data
                other_grad = grad.data * self.data
                
                # Handle broadcasting
                if self.shape != grad.shape:
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(self.shape) or self.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        self_grad = np.sum(self_grad, axis=tuple(axes_to_sum), keepdims=True)
                    self_grad = self_grad.reshape(self.shape)
                
                if other.shape != grad.shape:
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(other.shape) or other.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        other_grad = np.sum(other_grad, axis=tuple(axes_to_sum), keepdims=True)
                    other_grad = other_grad.reshape(other.shape)
                
                if self.requires_grad:
                    self.backward(Tensor(self_grad))
                if other.requires_grad:
                    other.backward(Tensor(other_grad))
            
            result._grad_fn = grad_fn
            result._children = [self, other]
        
        return result
    
    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        """Reverse multiplication."""
        return self.__mul__(other)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            return NotImplemented
        
        result = Tensor(np.matmul(self.data, other.data), 
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def grad_fn(grad):
                if self.requires_grad:
                    self_grad = np.matmul(grad.data, other.data.T)
                    self.backward(Tensor(self_grad))
                if other.requires_grad:
                    other_grad = np.matmul(self.data.T, grad.data)
                    other.backward(Tensor(other_grad))
            
            result._grad_fn = grad_fn
            result._children = [self, other]
        
        return result
    
    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Division operation."""
        if isinstance(other, (int, float)):
            other = Tensor(other)
        elif not isinstance(other, Tensor):
            return NotImplemented
        
        result = Tensor(self.data / other.data, 
                       requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def grad_fn(grad):
                self_grad = grad.data / other.data
                other_grad = -grad.data * self.data / (other.data ** 2)
                
                # Handle broadcasting
                if self.shape != grad.shape:
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(self.shape) or self.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        self_grad = np.sum(self_grad, axis=tuple(axes_to_sum), keepdims=True)
                    self_grad = self_grad.reshape(self.shape)
                
                if other.shape != grad.shape:
                    axes_to_sum = []
                    for i in range(len(grad.shape)):
                        if i >= len(other.shape) or other.shape[-(i+1)] == 1:
                            axes_to_sum.append(len(grad.shape) - i - 1)
                    if axes_to_sum:
                        other_grad = np.sum(other_grad, axis=tuple(axes_to_sum), keepdims=True)
                    other_grad = other_grad.reshape(other.shape)
                
                if self.requires_grad:
                    self.backward(Tensor(self_grad))
                if other.requires_grad:
                    other.backward(Tensor(other_grad))
            
            result._grad_fn = grad_fn
            result._children = [self, other]
        
        return result
    
    def __pow__(self, exponent: Union['Tensor', float, int]) -> 'Tensor':
        """Power operation."""
        if isinstance(exponent, (int, float)):
            exponent = Tensor(exponent)
        elif not isinstance(exponent, Tensor):
            return NotImplemented
        
        result = Tensor(np.power(self.data, exponent.data), 
                       requires_grad=self.requires_grad or exponent.requires_grad)
        
        if result.requires_grad:
            def grad_fn(grad):
                if self.requires_grad:
                    self_grad = grad.data * exponent.data * np.power(self.data, exponent.data - 1)
                    self.backward(Tensor(self_grad))
                if exponent.requires_grad:
                    exp_grad = grad.data * result.data * np.log(np.maximum(self.data, 1e-8))
                    exponent.backward(Tensor(exp_grad))
            
            result._grad_fn = grad_fn
            result._children = [self, exponent]
        
        return result
    
    # Comparison operations
    def __eq__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Equality comparison."""
        if isinstance(other, Tensor):
            return self.data == other.data
        return self.data == other
    
    def __ne__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Inequality comparison."""
        return ~self.__eq__(other)
    
    def __lt__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Less than comparison."""
        if isinstance(other, Tensor):
            return self.data < other.data
        return self.data < other
    
    def __le__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Less than or equal comparison."""
        if isinstance(other, Tensor):
            return self.data <= other.data
        return self.data <= other
    
    def __gt__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Greater than comparison."""
        if isinstance(other, Tensor):
            return self.data > other.data
        return self.data > other
    
    def __ge__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Greater than or equal comparison."""
        if isinstance(other, Tensor):
            return self.data >= other.data
        return self.data >= other
    
    # Indexing
    def __getitem__(self, key) -> 'Tensor':
        """Get item/slice from tensor."""
        result = Tensor(self.data[key], requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def grad_fn(grad):
                full_grad = np.zeros_like(self.data)
                full_grad[key] = grad.data
                self.backward(Tensor(full_grad))
            
            result._grad_fn = grad_fn
            result._children = [self]
        
        return result
    
    def __setitem__(self, key, value: Union['Tensor', float, int]):
        """Set item/slice in tensor."""
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    # String representation
    def __repr__(self) -> str:
        """String representation of the tensor."""
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_str})"
    
    def __str__(self) -> str:
        """String representation of the tensor."""
        return str(self.data)


# Utility functions for tensor operations
def zeros(*shape, requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor filled with zeros."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor filled with ones."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor with random normal distribution."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.randn(*shape).astype(dtype or np.float32), requires_grad=requires_grad)


def rand(*shape, requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor with random uniform distribution [0, 1)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.rand(*shape).astype(dtype or np.float32), requires_grad=requires_grad)


def eye(n: int, requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create an identity matrix."""
    return Tensor(np.eye(n, dtype=dtype), requires_grad=requires_grad)


def arange(start: int, stop: Optional[int] = None, step: int = 1, 
           requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor with evenly spaced values."""
    if stop is None:
        stop = start
        start = 0
    return Tensor(np.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad)