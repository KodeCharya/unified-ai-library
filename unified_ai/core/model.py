"""
Model class for the Unified AI Framework.
Provides a Keras-like interface for building and training neural networks.
"""

import numpy as np
from typing import List, Optional, Union, Callable, Dict, Any, Tuple
from .tensor import Tensor
from .layers import Layer
from .optimizer import Optimizer, Adam
from ..utils.losses import Loss, MSELoss
from ..utils.metrics import Metric, accuracy_score
from ..callbacks.early_stopping import EarlyStopping
from ..callbacks.model_checkpoint import ModelCheckpoint
from tqdm import tqdm
import time


class Model:
    """
    A neural network model that can be compiled and trained.
    
    This class provides a high-level interface for building, compiling, and training
    neural networks, similar to Keras models.
    """
    
    def __init__(self, layers: Optional[List[Layer]] = None, name: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            layers: List of layers to add to the model
            name: Name of the model
        """
        self.name = name or "Model"
        self.layers = layers or []
        self.built = False
        self.compiled = False
        
        # Training configuration
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        
        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'metrics': {},
            'val_metrics': {}
        }
    
    def add(self, layer: Layer):
        """Add a layer to the model."""
        self.layers.append(layer)
        self.built = False
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the model by building all layers."""
        if self.built:
            return
        
        current_shape = input_shape
        for layer in self.layers:
            layer.build(current_shape)
            current_shape = layer.output_shape
        
        self.built = True
        self.output_shape = current_shape
    
    def compile(self, optimizer: Union[Optimizer, str] = 'adam',
                loss: Union[Loss, str] = 'mse',
                metrics: Optional[List[Union[Metric, str]]] = None):
        """
        Compile the model for training.
        
        Args:
            optimizer: Optimizer instance or name
            loss: Loss function instance or name
            metrics: List of metrics to track
        """
        # Set optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                self.optimizer = Adam()
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer
        
        # Set loss function
        if isinstance(loss, str):
            if loss.lower() == 'mse':
                self.loss_fn = MSELoss()
            else:
                raise ValueError(f"Unknown loss function: {loss}")
        else:
            self.loss_fn = loss
        
        # Set metrics
        self.metrics = []
        if metrics:
            for metric in metrics:
                if isinstance(metric, str):
                    if metric.lower() == 'accuracy':
                        self.metrics.append(accuracy_score)
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                else:
                    self.metrics.append(metric)
        
        self.compiled = True
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Forward pass through the model."""
        if not self.built:
            self.build(x.shape)
        
        output = x
        for layer in self.layers:
            output = layer(output, training=training)
        
        return output
    
    def predict(self, x: Union[np.ndarray, Tensor], batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions for input samples.
        
        Args:
            x: Input data
            batch_size: Batch size for prediction
            
        Returns:
            Predictions as numpy array
        """
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        
        predictions = []
        num_samples = x.shape[0]
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_x = x[i:batch_end]
            
            batch_pred = self.forward(batch_x, training=False)
            predictions.append(batch_pred.data)
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(self, x: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor],
                 batch_size: int = 32, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            x: Input data
            y: Target data
            batch_size: Batch size for evaluation
            verbose: Whether to print progress
            
        Returns:
            Dictionary of metric values
        """
        if not self.compiled:
            raise RuntimeError("Model must be compiled before evaluation")
        
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if isinstance(y, np.ndarray):
            y = Tensor(y)
        
        total_loss = 0.0
        metric_values = {metric.__name__ if hasattr(metric, '__name__') else str(metric): 0.0 
                        for metric in self.metrics}
        num_batches = 0
        num_samples = x.shape[0]
        
        if verbose:
            pbar = tqdm(range(0, num_samples, batch_size), desc="Evaluating")
        else:
            pbar = range(0, num_samples, batch_size)
        
        for i in pbar:
            batch_end = min(i + batch_size, num_samples)
            batch_x = x[i:batch_end]
            batch_y = y[i:batch_end]
            
            # Forward pass
            predictions = self.forward(batch_x, training=False)
            
            # Calculate loss
            loss = self.loss_fn(batch_y, predictions)
            total_loss += loss.item()
            
            # Calculate metrics
            for metric in self.metrics:
                metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
                metric_value = metric(batch_y.data, predictions.data)
                metric_values[metric_name] += metric_value
            
            num_batches += 1
        
        # Average the results
        avg_loss = total_loss / num_batches
        for metric_name in metric_values:
            metric_values[metric_name] /= num_batches
        
        results = {'loss': avg_loss}
        results.update(metric_values)
        
        if verbose:
            print(f"Evaluation - Loss: {avg_loss:.4f}", end="")
            for metric_name, value in metric_values.items():
                print(f", {metric_name}: {value:.4f}", end="")
            print()
        
        return results
    
    def fit(self, x: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor],
            batch_size: int = 32, epochs: int = 1,
            validation_data: Optional[Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]] = None,
            callbacks: Optional[List] = None, verbose: bool = True,
            shuffle: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            x: Training input data
            y: Training target data
            batch_size: Batch size
            epochs: Number of epochs
            validation_data: Validation data tuple (x_val, y_val)
            callbacks: List of callbacks
            verbose: Whether to print training progress
            shuffle: Whether to shuffle training data
            
        Returns:
            Training history
        """
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training")
        
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if isinstance(y, np.ndarray):
            y = Tensor(y)
        
        # Prepare validation data
        val_x, val_y = None, None
        if validation_data:
            val_x, val_y = validation_data
            if isinstance(val_x, np.ndarray):
                val_x = Tensor(val_x)
            if isinstance(val_y, np.ndarray):
                val_y = Tensor(val_y)
        
        # Initialize callbacks
        callbacks = callbacks or []
        for callback in callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin()
        
        num_samples = x.shape[0]
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle data if requested
            if shuffle:
                indices = np.random.permutation(num_samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
            else:
                x_shuffled, y_shuffled = x, y
            
            # Training phase
            total_loss = 0.0
            metric_values = {metric.__name__ if hasattr(metric, '__name__') else str(metric): 0.0 
                           for metric in self.metrics}
            num_batches = 0
            
            if verbose:
                pbar = tqdm(range(0, num_samples, batch_size), 
                          desc=f"Epoch {epoch+1}/{epochs}")
            else:
                pbar = range(0, num_samples, batch_size)
            
            for i in pbar:
                batch_end = min(i + batch_size, num_samples)
                batch_x = x_shuffled[i:batch_end]
                batch_y = y_shuffled[i:batch_end]
                
                # Zero gradients
                self._zero_gradients()
                
                # Forward pass
                predictions = self.forward(batch_x, training=True)
                
                # Calculate loss
                loss = self.loss_fn(batch_y, predictions)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step(self._get_parameters())
                
                # Track metrics
                total_loss += loss.item()
                for metric in self.metrics:
                    metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
                    metric_value = metric(batch_y.data, predictions.data)
                    metric_values[metric_name] += metric_value
                
                num_batches += 1
                
                if verbose:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Average training metrics
            avg_train_loss = total_loss / num_batches
            avg_train_metrics = {name: value / num_batches for name, value in metric_values.items()}
            
            # Validation phase
            val_results = {}
            if validation_data:
                val_results = self.evaluate(val_x, val_y, batch_size, verbose=False)
            
            # Update history
            self.history['loss'].append(avg_train_loss)
            if validation_data:
                self.history['val_loss'].append(val_results['loss'])
            
            for metric_name, value in avg_train_metrics.items():
                if metric_name not in self.history['metrics']:
                    self.history['metrics'][metric_name] = []
                self.history['metrics'][metric_name].append(value)
                
                if validation_data and metric_name in val_results:
                    if metric_name not in self.history['val_metrics']:
                        self.history['val_metrics'][metric_name] = []
                    self.history['val_metrics'][metric_name].append(val_results[metric_name])
            
            # Print epoch results
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_train_loss:.4f}", end="")
                
                for metric_name, value in avg_train_metrics.items():
                    print(f" - {metric_name}: {value:.4f}", end="")
                
                if validation_data:
                    print(f" - val_loss: {val_results['loss']:.4f}", end="")
                    for metric_name, value in val_results.items():
                        if metric_name != 'loss':
                            print(f" - val_{metric_name}: {value:.4f}", end="")
                print()
            
            # Call callbacks
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    logs = {
                        'loss': avg_train_loss,
                        **avg_train_metrics
                    }
                    if validation_data:
                        logs.update({f'val_{k}': v for k, v in val_results.items()})
                    
                    callback.on_epoch_end(epoch, logs)
                    
                    # Check for early stopping
                    if hasattr(callback, 'stop_training') and callback.stop_training:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Call training end callbacks
        for callback in callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end()
        
        return self.history
    
    def _zero_gradients(self):
        """Zero all parameter gradients."""
        for layer in self.layers:
            for param in layer.get_weights():
                if param.requires_grad:
                    param.zero_grad()
    
    def _get_parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.get_weights())
        return parameters
    
    def save_weights(self, filepath: str):
        """Save model weights to file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            layer_weights = layer.get_weights()
            if layer_weights:
                weights[f'layer_{i}'] = [w.data for w in layer_weights]
        
        np.savez(filepath, **weights)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """Load model weights from file."""
        weights_data = np.load(filepath)
        
        for i, layer in enumerate(self.layers):
            layer_key = f'layer_{i}'
            if layer_key in weights_data.files:
                layer_weights = [Tensor(w, requires_grad=True) for w in weights_data[layer_key]]
                layer.set_weights(layer_weights)
        
        print(f"Model weights loaded from {filepath}")
    
    def summary(self):
        """Print model summary."""
        print(f"Model: {self.name}")
        print("=" * 65)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 65)
        
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.name} ({layer.__class__.__name__})"
            output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else "Unknown"
            
            # Count parameters
            layer_params = 0
            for param in layer.get_weights():
                layer_params += param.size
            
            print(f"{layer_name:<25} {output_shape:<20} {layer_params:<10}")
            
            total_params += layer_params
            if layer.trainable:
                trainable_params += layer_params
        
        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 65)
    
    def __call__(self, x: Tensor, training: bool = True) -> Tensor:
        """Make the model callable."""
        return self.forward(x, training)


# Functional API support
class Sequential(Model):
    """
    Sequential model for linear stack of layers.
    
    This is a convenience class that inherits from Model and provides
    a simple way to build models layer by layer.
    """
    
    def __init__(self, layers: Optional[List[Layer]] = None, name: Optional[str] = None):
        """
        Initialize Sequential model.
        
        Args:
            layers: List of layers
            name: Model name
        """
        super().__init__(layers, name or "Sequential")