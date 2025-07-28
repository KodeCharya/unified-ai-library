"""
Automatic differentiation system for the Unified AI Framework.
Provides computational graph construction and gradient computation.
"""

import numpy as np
from typing import Set, List, Callable, Optional, Any, Union
from .tensor import Tensor


class AutoDiff:
    """
    Automatic differentiation engine.
    
    This class provides utilities for building computational graphs
    and computing gradients automatically.
    """
    
    def __init__(self):
        """Initialize the autodiff engine."""
        self.tape = []
        self.recording = False
    
    def __enter__(self):
        """Enter gradient tape context."""
        self.recording = True
        self.tape = []
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit gradient tape context."""
        self.recording = False
    
    def watch(self, tensor: Tensor):
        """Watch a tensor for gradient computation."""
        if not tensor.requires_grad:
            tensor.requires_grad = True
            tensor.grad = np.zeros_like(tensor.data)
    
    def gradient(self, target: Tensor, sources: List[Tensor]) -> List[Tensor]:
        """
        Compute gradients of target with respect to sources.
        
        Args:
            target: Target tensor to differentiate
            sources: List of source tensors
            
        Returns:
            List of gradient tensors
        """
        # Ensure target is scalar
        if target.data.size != 1:
            raise ValueError("Target must be a scalar tensor")
        
        # Initialize gradients
        for source in sources:
            if source.grad is None:
                source.grad = np.zeros_like(source.data)
            else:
                source.grad.fill(0)
        
        # Backward pass
        target.backward()
        
        # Collect gradients
        gradients = []
        for source in sources:
            if source.grad is not None:
                gradients.append(Tensor(source.grad.copy()))
            else:
                gradients.append(Tensor(np.zeros_like(source.data)))
        
        return gradients
    
    def jacobian(self, outputs: List[Tensor], inputs: List[Tensor]) -> List[List[Tensor]]:
        """
        Compute Jacobian matrix.
        
        Args:
            outputs: List of output tensors
            inputs: List of input tensors
            
        Returns:
            Jacobian matrix as nested list of tensors
        """
        jacobian_matrix = []
        
        for output in outputs:
            output_grads = []
            for input_tensor in inputs:
                # Reset gradients
                for inp in inputs:
                    if inp.grad is not None:
                        inp.grad.fill(0)
                
                # Compute gradient
                if output.data.size == 1:
                    output.backward()
                    if input_tensor.grad is not None:
                        output_grads.append(Tensor(input_tensor.grad.copy()))
                    else:
                        output_grads.append(Tensor(np.zeros_like(input_tensor.data)))
                else:
                    # For non-scalar outputs, compute gradient for each element
                    grad_list = []
                    flat_output = output.reshape(-1)
                    for i in range(flat_output.shape[0]):
                        # Reset gradients
                        for inp in inputs:
                            if inp.grad is not None:
                                inp.grad.fill(0)
                        
                        # Create one-hot gradient
                        grad = np.zeros_like(flat_output.data)
                        grad[i] = 1.0
                        flat_output.backward(Tensor(grad))
                        
                        if input_tensor.grad is not None:
                            grad_list.append(input_tensor.grad.copy())
                        else:
                            grad_list.append(np.zeros_like(input_tensor.data))
                    
                    # Reshape gradient to match output shape
                    grad_array = np.array(grad_list).reshape(output.shape + input_tensor.shape)
                    output_grads.append(Tensor(grad_array))
            
            jacobian_matrix.append(output_grads)
        
        return jacobian_matrix
    
    def hessian(self, output: Tensor, inputs: List[Tensor]) -> List[List[Tensor]]:
        """
        Compute Hessian matrix (second derivatives).
        
        Args:
            output: Output tensor (must be scalar)
            inputs: List of input tensors
            
        Returns:
            Hessian matrix as nested list of tensors
        """
        if output.data.size != 1:
            raise ValueError("Output must be a scalar tensor")
        
        # First, compute first derivatives
        first_grads = self.gradient(output, inputs)
        
        # Then compute second derivatives
        hessian_matrix = []
        for i, first_grad in enumerate(first_grads):
            hessian_row = []
            for j, input_tensor in enumerate(inputs):
                if first_grad.data.size == 1:
                    second_grads = self.gradient(first_grad, [input_tensor])
                    hessian_row.append(second_grads[0])
                else:
                    # For vector first derivatives, compute Hessian for each element
                    second_grad_list = []
                    flat_first_grad = first_grad.reshape(-1)
                    for k in range(flat_first_grad.shape[0]):
                        element_grad = flat_first_grad[k]
                        if element_grad.data.size == 1:
                            second_grads = self.gradient(element_grad, [input_tensor])
                            second_grad_list.append(second_grads[0].data)
                        else:
                            second_grad_list.append(np.zeros_like(input_tensor.data))
                    
                    hessian_element = np.array(second_grad_list).reshape(
                        first_grad.shape + input_tensor.shape
                    )
                    hessian_row.append(Tensor(hessian_element))
            
            hessian_matrix.append(hessian_row)
        
        return hessian_matrix


# Gradient checking utilities
def numerical_gradient(func: Callable, inputs: List[Tensor], h: float = 1e-5) -> List[Tensor]:
    """
    Compute numerical gradients for gradient checking.
    
    Args:
        func: Function to differentiate
        inputs: List of input tensors
        h: Step size for numerical differentiation
        
    Returns:
        List of numerical gradient tensors
    """
    gradients = []
    
    for input_tensor in inputs:
        grad = np.zeros_like(input_tensor.data)
        
        # Flatten for easier iteration
        flat_input = input_tensor.data.flatten()
        flat_grad = grad.flatten()
        
        for i in range(len(flat_input)):
            # Forward difference
            flat_input[i] += h
            input_tensor.data = flat_input.reshape(input_tensor.shape)
            f_plus = func(inputs).item()
            
            # Backward difference
            flat_input[i] -= 2 * h
            input_tensor.data = flat_input.reshape(input_tensor.shape)
            f_minus = func(inputs).item()
            
            # Restore original value
            flat_input[i] += h
            input_tensor.data = flat_input.reshape(input_tensor.shape)
            
            # Compute numerical gradient
            flat_grad[i] = (f_plus - f_minus) / (2 * h)
        
        gradients.append(Tensor(grad))
    
    return gradients


def gradient_check(func: Callable, inputs: List[Tensor], 
                  analytical_grads: List[Tensor], h: float = 1e-5,
                  tolerance: float = 1e-6) -> bool:
    """
    Check analytical gradients against numerical gradients.
    
    Args:
        func: Function to differentiate
        inputs: List of input tensors
        analytical_grads: List of analytical gradient tensors
        h: Step size for numerical differentiation
        tolerance: Tolerance for gradient checking
        
    Returns:
        True if gradients match within tolerance
    """
    numerical_grads = numerical_gradient(func, inputs, h)
    
    for i, (analytical, numerical) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.abs(analytical.data - numerical.data)
        max_diff = np.max(diff)
        
        if max_diff > tolerance:
            print(f"Gradient check failed for input {i}")
            print(f"Max difference: {max_diff}")
            print(f"Analytical gradient: {analytical.data}")
            print(f"Numerical gradient: {numerical.data}")
            return False
    
    print("Gradient check passed!")
    return True


# Higher-order derivatives
def grad(func: Callable, inputs: List[Tensor]) -> Callable:
    """
    Create a function that computes gradients.
    
    Args:
        func: Function to differentiate
        inputs: List of input tensors
        
    Returns:
        Function that computes gradients
    """
    def grad_func(*args):
        # Ensure inputs require gradients
        for inp in inputs:
            inp.requires_grad = True
            inp.grad = np.zeros_like(inp.data)
        
        # Compute function
        output = func(*args)
        
        # Compute gradients
        output.backward()
        
        # Return gradients
        return [Tensor(inp.grad.copy()) for inp in inputs]
    
    return grad_func


def value_and_grad(func: Callable, inputs: List[Tensor]) -> Callable:
    """
    Create a function that computes both value and gradients.
    
    Args:
        func: Function to differentiate
        inputs: List of input tensors
        
    Returns:
        Function that computes value and gradients
    """
    def value_and_grad_func(*args):
        # Ensure inputs require gradients
        for inp in inputs:
            inp.requires_grad = True
            inp.grad = np.zeros_like(inp.data)
        
        # Compute function
        output = func(*args)
        
        # Compute gradients
        output.backward()
        
        # Return value and gradients
        gradients = [Tensor(inp.grad.copy()) for inp in inputs]
        return output, gradients
    
    return value_and_grad_func


# Functional transformations
def jit(func: Callable) -> Callable:
    """
    Just-in-time compilation placeholder.
    
    In a full implementation, this would compile the function for faster execution.
    For now, it just returns the original function.
    
    Args:
        func: Function to compile
        
    Returns:
        Compiled function (currently just the original function)
    """
    # Placeholder for JIT compilation
    # In a real implementation, this would use techniques like:
    # - Tracing the computational graph
    # - Optimizing the graph (fusion, elimination, etc.)
    # - Generating optimized code
    return func


def vmap(func: Callable, in_axes: Union[int, List[int]] = 0) -> Callable:
    """
    Vectorizing map placeholder.
    
    This would automatically vectorize a function over specified axes.
    
    Args:
        func: Function to vectorize
        in_axes: Axes to vectorize over
        
    Returns:
        Vectorized function
    """
    def vmapped_func(*args):
        # Placeholder implementation
        # In a real implementation, this would:
        # - Determine the batch dimension
        # - Apply the function to each element in the batch
        # - Stack the results
        
        if isinstance(in_axes, int):
            batch_size = args[0].shape[in_axes]
            results = []
            
            for i in range(batch_size):
                batch_args = []
                for arg in args:
                    if in_axes < len(arg.shape):
                        batch_args.append(arg[i] if in_axes == 0 else 
                                        arg.transpose()[i].transpose())
                    else:
                        batch_args.append(arg)
                
                result = func(*batch_args)
                results.append(result)
            
            # Stack results
            if results:
                stacked_data = np.stack([r.data for r in results], axis=in_axes)
                return Tensor(stacked_data, requires_grad=any(r.requires_grad for r in results))
            else:
                return func(*args)
        else:
            # Multiple axes case - more complex implementation needed
            return func(*args)
    
    return vmapped_func


# Context managers for gradient computation
class no_grad:
    """Context manager to disable gradient computation."""
    
    def __init__(self):
        self.prev_state = {}
    
    def __enter__(self):
        # This would disable gradient computation globally
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state
        pass


class enable_grad:
    """Context manager to enable gradient computation."""
    
    def __init__(self):
        self.prev_state = {}
    
    def __enter__(self):
        # This would enable gradient computation globally
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state
        pass