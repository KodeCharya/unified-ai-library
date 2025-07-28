"""
Model checkpoint callback for the Unified AI Framework.
Saves the model at specified intervals or when performance improves.
"""

import os
import numpy as np
from typing import Optional, Dict, Any


class ModelCheckpoint:
    """
    Save the model after every epoch or when performance improves.
    
    This callback saves the model's weights (and optionally the full model)
    at some interval or when the model achieves better performance.
    """
    
    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 verbose: int = 0, save_best_only: bool = False,
                 save_weights_only: bool = False, mode: str = 'auto',
                 save_freq: str = 'epoch', period: int = 1):
        """
        Initialize ModelCheckpoint callback.
        
        Args:
            filepath: String or PathLike, path to save the model file
            monitor: Quantity to monitor
            verbose: Verbosity mode (0 = silent, 1 = progress messages)
            save_best_only: If True, only save when the model is considered the "best"
            save_weights_only: If True, then only the model's weights will be saved
            mode: One of {'auto', 'min', 'max'}. If save_best_only=True, the decision
                  to overwrite the current save file is made based on either the
                  maximization or the minimization of the monitored quantity
            save_freq: 'epoch' or integer. When using 'epoch', the callback saves
                      the model after each epoch. When using integer, the callback
                      saves the model at end of this many batches
            period: Interval (number of epochs) between checkpoints (deprecated, use save_freq)
        """
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.period = period
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if mode not in ['auto', 'min', 'max']:
            raise ValueError(f"Mode {mode} is unknown")
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:  # mode == 'auto'
            if 'acc' in monitor or monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
        
        self.epochs_since_last_save = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        self.epochs_since_last_save = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        logs = logs or {}
        self.epochs_since_last_save += 1
        
        if self.save_freq == 'epoch':
            if self.epochs_since_last_save >= self.period:
                self._save_model(epoch, logs)
                self.epochs_since_last_save = 0
    
    def _save_model(self, epoch: int, logs: Dict[str, Any]):
        """Save the model."""
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                if self.verbose > 0:
                    print(f"Can save best model only with {self.monitor} available, "
                          f"skipping.")
                return
            
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved from "
                          f"{self.best:.5f} to {current:.5f}, saving model to {filepath}")
                self.best = current
                self._do_save(filepath)
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")
        else:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: saving model to {filepath}")
            self._do_save(filepath)
    
    def _do_save(self, filepath: str):
        """Actually save the model (placeholder implementation)."""
        # This would be implemented to save the actual model
        # For now, create a placeholder file
        try:
            if self.save_weights_only:
                # Save only weights
                # model.save_weights(filepath)
                with open(filepath, 'w') as f:
                    f.write("Model weights placeholder")
            else:
                # Save full model
                # model.save(filepath)
                with open(filepath, 'w') as f:
                    f.write("Full model placeholder")
        except Exception as e:
            if self.verbose > 0:
                print(f"Error saving model: {e}")


class BackupAndRestore:
    """
    Backup and restore callback for fault tolerance.
    
    This callback backs up the model and optimizer state at regular intervals
    and can restore from the latest backup in case of interruption.
    """
    
    def __init__(self, backup_dir: str, save_freq: int = 5,
                 delete_checkpoint: bool = True):
        """
        Initialize BackupAndRestore callback.
        
        Args:
            backup_dir: Directory to save backup files
            save_freq: Frequency (in epochs) to save backups
            delete_checkpoint: Whether to delete checkpoint after successful training
        """
        self.backup_dir = backup_dir
        self.save_freq = save_freq
        self.delete_checkpoint = delete_checkpoint
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        self.backup_file = os.path.join(backup_dir, 'backup.npz')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        if (epoch + 1) % self.save_freq == 0:
            self._save_backup(epoch, logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        if self.delete_checkpoint and os.path.exists(self.backup_file):
            os.remove(self.backup_file)
    
    def _save_backup(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save backup of model and training state."""
        backup_data = {
            'epoch': epoch,
            'logs': logs or {},
            # Add model weights, optimizer state, etc.
        }
        
        try:
            np.savez(self.backup_file, **backup_data)
        except Exception as e:
            print(f"Error saving backup: {e}")
    
    def restore_from_backup(self):
        """Restore model and training state from backup."""
        if not os.path.exists(self.backup_file):
            return None
        
        try:
            backup_data = np.load(self.backup_file, allow_pickle=True)
            return {
                'epoch': int(backup_data['epoch']),
                'logs': backup_data['logs'].item() if 'logs' in backup_data else {}
            }
        except Exception as e:
            print(f"Error loading backup: {e}")
            return None


class CSVLogger:
    """
    Callback that streams epoch results to a CSV file.
    
    This callback writes training metrics to a CSV file after each epoch.
    """
    
    def __init__(self, filename: str, separator: str = ',', append: bool = False):
        """
        Initialize CSVLogger callback.
        
        Args:
            filename: Filename of the CSV file
            separator: String used to separate elements in the CSV file
            append: Whether to append if file exists (True) or overwrite (False)
        """
        self.filename = filename
        self.separator = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        
        self.csv_file = open(self.filename, mode, newline='')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, (int, float)) or is_zero_dim_ndarray:
                return k
            else:
                return '"' + str(k).replace('"', '""') + '"'
        
        if self.keys is None:
            self.keys = sorted(logs.keys())
        
        if not self.writer:
            import csv
            class CustomDialect(csv.excel):
                delimiter = self.separator
            
            self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch'] + self.keys,
                                       dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        
        row_dict = {'epoch': epoch}
        row_dict.update((key, handle_value(logs.get(key, 'NA'))) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        self.csv_file.close()
        self.writer = None


class TensorBoard:
    """
    TensorBoard callback for visualization.
    
    This callback writes training metrics and other data for visualization in TensorBoard.
    Note: This is a simplified implementation - a full version would integrate with
    actual TensorBoard logging.
    """
    
    def __init__(self, log_dir: str = './logs', histogram_freq: int = 0,
                 write_graph: bool = True, write_images: bool = False,
                 update_freq: str = 'epoch', profile_batch: int = 2):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory where to save the log files
            histogram_freq: Frequency (in epochs) at which to compute activation histograms
            write_graph: Whether to visualize the graph in TensorBoard
            write_images: Whether to write model weights to visualize as image in TensorBoard
            update_freq: 'batch' or 'epoch' or integer
            profile_batch: Profile the batch to sample compute characteristics
        """
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        self.update_freq = update_freq
        self.profile_batch = profile_batch
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(log_dir, 'training_log.txt')
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        with open(self.log_file, 'w') as f:
            f.write("Training started\n")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Write metrics to log file
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}: ")
            for key, value in logs.items():
                f.write(f"{key}={value:.4f} ")
            f.write("\n")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        with open(self.log_file, 'a') as f:
            f.write("Training completed\n")