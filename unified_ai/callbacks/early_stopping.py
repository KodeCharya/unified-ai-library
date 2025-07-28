"""
Early stopping callback for the Unified AI Framework.
Stops training when a monitored metric has stopped improving.
"""

import numpy as np
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.
    
    This callback monitors a quantity and stops training when it stops improving.
    """
    
    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0,
                 patience: int = 0, verbose: int = 0, mode: str = 'auto',
                 baseline: Optional[float] = None, restore_best_weights: bool = False):
        """
        Initialize EarlyStopping callback.
        
        Args:
            monitor: Quantity to be monitored
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            patience: Number of epochs with no improvement after which training will be stopped
            verbose: Verbosity mode (0 = silent, 1 = update messages)
            mode: One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when
                  the quantity monitored has stopped decreasing; in 'max' mode it will
                  stop when the quantity monitored has stopped increasing
            baseline: Baseline value for the monitored quantity
            restore_best_weights: Whether to restore model weights from the epoch with
                                the best value of the monitored quantity
        """
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        if mode not in ['auto', 'min', 'max']:
            raise ValueError(f"Mode {mode} is unknown, please use one of 'auto', 'min', 'max'")
        
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:  # mode == 'auto'
            if 'acc' in monitor or monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.best_weights = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                print(f"Early stopping conditioned on metric `{self.monitor}` "
                      f"which is not available. Available metrics are: {list(logs.keys())}")
            return
        
        if self.restore_best_weights and self.best_weights is None:
            # Save initial weights
            self.best_weights = self._get_model_weights()
        
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self._get_model_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print("Restoring model weights from the end of the best epoch.")
                    self._set_model_weights(self.best_weights)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")
    
    def _get_model_weights(self):
        """Get model weights (placeholder - would be implemented with actual model)."""
        # This would be implemented to get weights from the actual model
        # For now, return None as placeholder
        return None
    
    def _set_model_weights(self, weights):
        """Set model weights (placeholder - would be implemented with actual model)."""
        # This would be implemented to set weights on the actual model
        # For now, do nothing as placeholder
        pass


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates.
    """
    
    def __init__(self, monitor: str = 'val_loss', factor: float = 0.1,
                 patience: int = 10, verbose: int = 0, mode: str = 'auto',
                 min_delta: float = 1e-4, cooldown: int = 0, min_lr: float = 0):
        """
        Initialize ReduceLROnPlateau callback.
        
        Args:
            monitor: Quantity to be monitored
            factor: Factor by which the learning rate will be reduced
            patience: Number of epochs with no improvement after which learning rate will be reduced
            verbose: Verbosity mode (0 = silent, 1 = update messages)
            mode: One of {'auto', 'min', 'max'}
            min_delta: Threshold for measuring the new optimum
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: Lower bound on the learning rate
        """
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        
        if mode not in ['auto', 'min', 'max']:
            raise ValueError(f"Mode {mode} is unknown")
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:  # mode == 'auto'
            if 'acc' in monitor:
                self.monitor_op = np.greater
                self.min_delta *= 1
            else:
                self.monitor_op = np.less
                self.min_delta *= -1
        
        self.cooldown_counter = 0
        self.wait = 0
        self.best = None
        self.lr_epsilon = min_lr * 1e-4
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        self.wait = 0
        self.cooldown_counter = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                print(f"ReduceLROnPlateau conditioned on metric `{self.monitor}` "
                      f"which is not available. Available metrics are: {list(logs.keys())}")
            return
        
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self._get_learning_rate()
                if old_lr > self.min_lr + self.lr_epsilon:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self._set_learning_rate(new_lr)
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing "
                              f"learning rate from {old_lr} to {new_lr}.")
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
    
    def in_cooldown(self) -> bool:
        """Check if we are in cooldown period."""
        return self.cooldown_counter > 0
    
    def _get_learning_rate(self) -> float:
        """Get current learning rate (placeholder)."""
        # This would be implemented to get LR from the actual optimizer
        return 0.001
    
    def _set_learning_rate(self, lr: float):
        """Set learning rate (placeholder)."""
        # This would be implemented to set LR on the actual optimizer
        pass


class LearningRateScheduler:
    """
    Learning rate scheduler callback.
    
    This callback allows you to schedule the learning rate using a custom function.
    """
    
    def __init__(self, schedule, verbose: int = 0):
        """
        Initialize LearningRateScheduler callback.
        
        Args:
            schedule: A function that takes an epoch index as input and returns a new learning rate
            verbose: Verbosity mode (0 = silent, 1 = update messages)
        """
        self.schedule = schedule
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError('LearningRateScheduler callback requires a model')
        
        lr = float(self.schedule(epoch))
        self._set_learning_rate(lr)
        
        if self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning rate to {lr}.")
    
    def _set_learning_rate(self, lr: float):
        """Set learning rate (placeholder)."""
        # This would be implemented to set LR on the actual optimizer
        pass