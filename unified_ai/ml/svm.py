"""
Support Vector Machine implementation for the Unified AI Framework.
Supports both classification and regression with various kernels.
"""

import numpy as np
from typing import Union, Optional, Callable
import warnings


class SVM:
    """
    Support Vector Machine for classification and regression.
    
    This implementation uses the Sequential Minimal Optimization (SMO) algorithm
    for training and supports various kernel functions.
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', degree: int = 3,
                 gamma: Union[str, float] = 'scale', coef0: float = 0.0,
                 tol: float = 1e-3, max_iter: int = 1000, random_state: Optional[int] = None):
        """
        Initialize SVM.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            degree: Degree of polynomial kernel
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
            coef0: Independent term in kernel function
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Model parameters (set during training)
        self.support_vectors_ = None
        self.support_ = None
        self.n_support_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.b = 0.0
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Validate parameters
        if C <= 0:
            raise ValueError("C must be positive")
        
        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        if tol <= 0:
            raise ValueError("tol must be positive")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Returns the instance itself
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # For binary classification, ensure labels are -1 and 1
        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            self.classes_ = unique_labels
            # Map labels to -1 and 1
            y_mapped = np.where(y == unique_labels[0], -1, 1)
        else:
            raise ValueError("This implementation only supports binary classification")
        
        # Set gamma if it's 'scale' or 'auto'
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self.gamma = 1.0 / (X.shape[1] * X.var())
            elif self.gamma == 'auto':
                self.gamma = 1.0 / X.shape[1]
        
        # Train using SMO algorithm
        self._train_smo(X, y_mapped)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Test data of shape (n_samples, n_features)
        
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Compute decision function
        decision = self.decision_function(X)
        
        # Convert to class labels
        predictions = np.where(decision >= 0, self.classes_[1], self.classes_[0])
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.
        
        Args:
            X: Test data of shape (n_samples, n_features)
        
        Returns:
            Decision function values of shape (n_samples,)
        """
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before computing decision function")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Compute kernel matrix between X and support vectors
        K = self._compute_kernel_matrix(X, self.support_vectors_)
        
        # Compute decision function: sum(alpha_i * y_i * K(x, x_i)) + b
        decision = np.dot(K, self.dual_coef_) + self.intercept_
        
        return decision
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
        
        Returns:
            Mean accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def _train_smo(self, X: np.ndarray, y: np.ndarray):
        """
        Train SVM using Sequential Minimal Optimization (SMO) algorithm.
        
        Args:
            X: Training data
            y: Training labels (-1 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize Lagrange multipliers
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        # Precompute kernel matrix for efficiency
        self.K = self._compute_kernel_matrix(X, X)
        
        # SMO main loop
        num_changed = 0
        examine_all = True
        
        for iteration in range(self.max_iter):
            num_changed = 0
            
            if examine_all:
                # Examine all examples
                for i in range(n_samples):
                    num_changed += self._examine_example(i, X, y)
            else:
                # Examine non-bound examples (0 < alpha < C)
                for i in range(n_samples):
                    if 0 < self.alpha[i] < self.C:
                        num_changed += self._examine_example(i, X, y)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            # Check for convergence
            if examine_all and num_changed == 0:
                break
        
        # Extract support vectors
        support_mask = self.alpha > self.tol
        self.support_ = np.where(support_mask)[0]
        self.support_vectors_ = X[support_mask]
        self.dual_coef_ = (self.alpha * y)[support_mask]
        self.intercept_ = self.b
        self.n_support_ = len(self.support_)
    
    def _examine_example(self, i2: int, X: np.ndarray, y: np.ndarray) -> int:
        """
        Examine example i2 and try to find a second example to optimize with.
        
        Args:
            i2: Index of second example
            X: Training data
            y: Training labels
        
        Returns:
            1 if optimization occurred, 0 otherwise
        """
        y2 = y[i2]
        alpha2 = self.alpha[i2]
        E2 = self._compute_error(i2, X, y)
        r2 = E2 * y2
        
        # Check KKT conditions
        if ((r2 < -self.tol and alpha2 < self.C) or 
            (r2 > self.tol and alpha2 > 0)):
            
            # Try to find i1 using second choice heuristic
            non_bound_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            
            if len(non_bound_indices) > 1:
                # Choose i1 to maximize |E1 - E2|
                errors = np.array([self._compute_error(i, X, y) for i in non_bound_indices])
                if E2 > 0:
                    i1 = non_bound_indices[np.argmin(errors)]
                else:
                    i1 = non_bound_indices[np.argmax(errors)]
                
                if self._take_step(i1, i2, X, y):
                    return 1
            
            # Try all non-bound examples
            for i1 in non_bound_indices:
                if self._take_step(i1, i2, X, y):
                    return 1
            
            # Try all examples
            for i1 in range(len(X)):
                if self._take_step(i1, i2, X, y):
                    return 1
        
        return 0
    
    def _take_step(self, i1: int, i2: int, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Attempt to jointly optimize alpha[i1] and alpha[i2].
        
        Args:
            i1: Index of first example
            i2: Index of second example
            X: Training data
            y: Training labels
        
        Returns:
            True if optimization occurred, False otherwise
        """
        if i1 == i2:
            return False
        
        alpha1, alpha2 = self.alpha[i1], self.alpha[i2]
        y1, y2 = y[i1], y[i2]
        E1 = self._compute_error(i1, X, y)
        E2 = self._compute_error(i2, X, y)
        s = y1 * y2
        
        # Compute bounds L and H
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        
        if L == H:
            return False
        
        # Compute eta (second derivative of objective function)
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12
        
        if eta > 0:
            # Compute new alpha2
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            
            # Clip alpha2_new to [L, H]
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:
            # eta <= 0, compute objective function at endpoints
            f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.b) - s * alpha1 * k12 - alpha2 * k22
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * L1**2 * k11 + 0.5 * L**2 * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1**2 * k11 + 0.5 * H**2 * k22 + s * H * H1 * k12
            
            if Lobj < Hobj - self.tol:
                alpha2_new = L
            elif Lobj > Hobj + self.tol:
                alpha2_new = H
            else:
                alpha2_new = alpha2
        
        # Check if change is significant
        if abs(alpha2_new - alpha2) < self.tol * (alpha2_new + alpha2 + self.tol):
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)
        
        # Update threshold b
        b1 = E1 + y1 * (alpha1_new - alpha1) * k11 + y2 * (alpha2_new - alpha2) * k12 + self.b
        b2 = E2 + y1 * (alpha1_new - alpha1) * k12 + y2 * (alpha2_new - alpha2) * k22 + self.b
        
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # Update alphas
        self.alpha[i1] = alpha1_new
        self.alpha[i2] = alpha2_new
        
        return True
    
    def _compute_error(self, i: int, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute prediction error for example i.
        
        Args:
            i: Example index
            X: Training data
            y: Training labels
        
        Returns:
            Prediction error
        """
        prediction = np.sum(self.alpha * y * self.K[i, :]) + self.b
        return prediction - y[i]
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between two sets of points.
        
        Args:
            X1: First set of points of shape (n1, n_features)
            X2: Second set of points of shape (n2, n_features)
        
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'rbf':
            # Compute squared Euclidean distances
            X1_norm = np.sum(X1**2, axis=1, keepdims=True)
            X2_norm = np.sum(X2**2, axis=1, keepdims=True)
            distances_sq = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * distances_sq)
        
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, return parameters for sub-estimators too
        
        Returns:
            Parameter names mapped to their values
        """
        return {
            'C': self.C,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Estimator parameters
        
        Returns:
            self: Estimator instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        
        return self


class SVR(SVM):
    """
    Support Vector Regression.
    
    This is a specialized version of SVM for regression tasks.
    """
    
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel: str = 'rbf',
                 degree: int = 3, gamma: Union[str, float] = 'scale', coef0: float = 0.0,
                 tol: float = 1e-3, max_iter: int = 1000, random_state: Optional[int] = None):
        """
        Initialize SVR.
        
        Args:
            C: Regularization parameter
            epsilon: Epsilon in the epsilon-SVR model
            kernel: Kernel type
            degree: Degree of polynomial kernel
            gamma: Kernel coefficient
            coef0: Independent term in kernel function
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
            random_state: Random seed
        """
        super().__init__(C, kernel, degree, gamma, coef0, tol, max_iter, random_state)
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVR model.
        
        This is a simplified implementation. A full SVR would require
        a more complex optimization procedure.
        """
        # For now, use a simple approach
        # In a full implementation, this would use epsilon-insensitive loss
        super().fit(X, np.sign(y - np.mean(y)))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the SVR model.
        
        This is a simplified implementation.
        """
        # For now, return decision function values
        return self.decision_function(X)