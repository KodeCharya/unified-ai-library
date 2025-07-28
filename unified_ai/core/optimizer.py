"""
Optimizers for the Unified AI Framework.
Provides various optimization algorithms for training neural networks.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .tensor import Tensor


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    All optimizers should inherit from this class and implement the step method.
    """
    
    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize the optimizer.
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
        self.learning_rate = learning_rate
        self.iterations = 0
    
    @abstractmethod
    def step(self, parameters: List[Tensor]):
        """
        Perform a single optimization step.
        
        Args:
            parameters: List of parameters to optimize
        """
        pass
    
    def zero_grad(self, parameters: List[Tensor]):
        """Zero the gradients of all parameters."""
        for param in parameters:
            if param.requires_grad:
                param.zero_grad()
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'iterations': self.iterations
        }


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Implements SGD with optional momentum and weight decay.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 regularization)
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Momentum buffers
        self.velocity = {}
    
    def step(self, parameters: List[Tensor]):
        """Perform SGD optimization step."""
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                if i not in self.velocity:
                    self.velocity[i] = np.zeros_like(param.data)
                
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * self.velocity[i]
                else:
                    grad = self.velocity[i]
            
            # Update parameters
            param.data -= self.learning_rate * grad
        
        self.iterations += 1
    
    def get_config(self) -> Dict[str, Any]:
        """Get SGD configuration."""
        config = super().get_config()
        config.update({
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov
        })
        return config


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Implements the Adam algorithm with bias correction.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay (L2 regularization)
            amsgrad: Whether to use AMSGrad variant
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Moment estimates
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.v_hat_max = {}  # For AMSGrad
    
    def step(self, parameters: List[Tensor]):
        """Perform Adam optimization step."""
        self.iterations += 1
        
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Initialize moment estimates
            if i not in self.m:
                self.m[i] = np.zeros_like(param.data)
                self.v[i] = np.zeros_like(param.data)
                if self.amsgrad:
                    self.v_hat_max[i] = np.zeros_like(param.data)
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.iterations)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.iterations)
            
            # AMSGrad variant
            if self.amsgrad:
                self.v_hat_max[i] = np.maximum(self.v_hat_max[i], v_hat)
                v_hat = self.v_hat_max[i]
            
            # Update parameters
            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def get_config(self) -> Dict[str, Any]:
        """Get Adam configuration."""
        config = super().get_config()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad
        })
        return config


class RMSProp(Optimizer):
    """
    RMSProp optimizer.
    
    Implements the RMSProp algorithm.
    """
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9,
                 epsilon: float = 1e-8, weight_decay: float = 0.0,
                 momentum: float = 0.0, centered: bool = False):
        """
        Initialize RMSProp optimizer.
        
        Args:
            learning_rate: Learning rate
            rho: Smoothing constant
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay (L2 regularization)
            momentum: Momentum factor
            centered: Whether to use centered RMSProp
        """
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        
        # Running averages
        self.square_avg = {}
        self.momentum_buffer = {}
        if centered:
            self.grad_avg = {}
    
    def step(self, parameters: List[Tensor]):
        """Perform RMSProp optimization step."""
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Initialize running averages
            if i not in self.square_avg:
                self.square_avg[i] = np.zeros_like(param.data)
                if self.momentum > 0:
                    self.momentum_buffer[i] = np.zeros_like(param.data)
                if self.centered:
                    self.grad_avg[i] = np.zeros_like(param.data)
            
            # Update running average of squared gradients
            self.square_avg[i] = self.rho * self.square_avg[i] + (1 - self.rho) * (grad ** 2)
            
            if self.centered:
                # Update running average of gradients
                self.grad_avg[i] = self.rho * self.grad_avg[i] + (1 - self.rho) * grad
                avg = self.square_avg[i] - self.grad_avg[i] ** 2
            else:
                avg = self.square_avg[i]
            
            # Compute update
            if self.momentum > 0:
                self.momentum_buffer[i] = (self.momentum * self.momentum_buffer[i] + 
                                         grad / (np.sqrt(avg) + self.epsilon))
                param.data -= self.learning_rate * self.momentum_buffer[i]
            else:
                param.data -= self.learning_rate * grad / (np.sqrt(avg) + self.epsilon)
        
        self.iterations += 1
    
    def get_config(self) -> Dict[str, Any]:
        """Get RMSProp configuration."""
        config = super().get_config()
        config.update({
            'rho': self.rho,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'centered': self.centered
        })
        return config


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.
    
    Implements the AdaGrad algorithm.
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-10,
                 weight_decay: float = 0.0):
        """
        Initialize AdaGrad optimizer.
        
        Args:
            learning_rate: Learning rate
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Accumulated squared gradients
        self.sum_squares = {}
    
    def step(self, parameters: List[Tensor]):
        """Perform AdaGrad optimization step."""
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Initialize accumulated squared gradients
            if i not in self.sum_squares:
                self.sum_squares[i] = np.zeros_like(param.data)
            
            # Accumulate squared gradients
            self.sum_squares[i] += grad ** 2
            
            # Update parameters
            param.data -= (self.learning_rate / 
                          (np.sqrt(self.sum_squares[i]) + self.epsilon)) * grad
        
        self.iterations += 1
    
    def get_config(self) -> Dict[str, Any]:
        """Get AdaGrad configuration."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay
        })
        return config


class AdaDelta(Optimizer):
    """
    AdaDelta optimizer.
    
    Implements the AdaDelta algorithm.
    """
    
    def __init__(self, learning_rate: float = 1.0, rho: float = 0.9,
                 epsilon: float = 1e-6, weight_decay: float = 0.0):
        """
        Initialize AdaDelta optimizer.
        
        Args:
            learning_rate: Learning rate (usually set to 1.0)
            rho: Coefficient for computing running averages
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Running averages
        self.square_avg = {}
        self.acc_delta = {}
    
    def step(self, parameters: List[Tensor]):
        """Perform AdaDelta optimization step."""
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Initialize running averages
            if i not in self.square_avg:
                self.square_avg[i] = np.zeros_like(param.data)
                self.acc_delta[i] = np.zeros_like(param.data)
            
            # Update running average of squared gradients
            self.square_avg[i] = self.rho * self.square_avg[i] + (1 - self.rho) * (grad ** 2)
            
            # Compute update
            std = np.sqrt(self.acc_delta[i] + self.epsilon)
            delta = -(std / np.sqrt(self.square_avg[i] + self.epsilon)) * grad
            
            # Update running average of squared deltas
            self.acc_delta[i] = self.rho * self.acc_delta[i] + (1 - self.rho) * (delta ** 2)
            
            # Update parameters
            param.data += self.learning_rate * delta
        
        self.iterations += 1
    
    def get_config(self) -> Dict[str, Any]:
        """Get AdaDelta configuration."""
        config = super().get_config()
        config.update({
            'rho': self.rho,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay
        })
        return config


# Learning rate schedulers
class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: Optimizer):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """Update learning rate."""
        pass


class StepLR(LearningRateScheduler):
    """Step learning rate scheduler."""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        """
        Initialize StepLR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        if epoch % self.step_size == 0 and epoch > 0:
            self.optimizer.learning_rate *= self.gamma


class ExponentialLR(LearningRateScheduler):
    """Exponential learning rate scheduler."""
    
    def __init__(self, optimizer: Optimizer, gamma: float):
        """
        Initialize ExponentialLR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.gamma = gamma
        self.last_epoch = 0
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self.optimizer.learning_rate = self.base_lr * (self.gamma ** epoch)


class ReduceLROnPlateau(LearningRateScheduler):
    """Reduce learning rate when metric has stopped improving."""
    
    def __init__(self, optimizer: Optimizer, mode: str = 'min', factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4, min_lr: float = 0):
        """
        Initialize ReduceLROnPlateau scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            mode: 'min' or 'max' - whether to minimize or maximize the metric
            factor: Factor by which learning rate will be reduced
            patience: Number of epochs with no improvement after which LR will be reduced
            threshold: Threshold for measuring the new optimum
            min_lr: Lower bound on the learning rate
        """
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self._init_is_better()
    
    def _init_is_better(self):
        """Initialize comparison function."""
        if self.mode == 'min':
            self.mode_worse = float('inf')
        else:
            self.mode_worse = -float('inf')
    
    def _is_better(self, a, best):
        """Check if a is better than best."""
        if self.mode == 'min':
            return a < best - self.threshold
        else:
            return a > best + self.threshold
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """Update learning rate based on metrics."""
        if metrics is None:
            return
        
        if self.best is None:
            self.best = metrics
        elif self._is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            old_lr = self.optimizer.learning_rate
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.optimizer.learning_rate = new_lr
            self.num_bad_epochs = 0
            print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")