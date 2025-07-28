"""
K-Nearest Neighbors implementation for the Unified AI Framework.
Supports both classification and regression tasks.
"""

import numpy as np
from typing import Union, Optional, Callable
from collections import Counter
import warnings


class KNN:
    """
    K-Nearest Neighbors classifier and regressor.
    
    This implementation supports both classification and regression tasks
    using various distance metrics and weighting schemes.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 algorithm: str = 'auto', leaf_size: int = 30,
                 p: int = 2, metric: str = 'minkowski',
                 metric_params: Optional[dict] = None, n_jobs: Optional[int] = None):
        """
        Initialize KNN classifier/regressor.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function used in prediction ('uniform', 'distance')
            algorithm: Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Leaf size passed to BallTree or KDTree
            p: Parameter for the Minkowski metric
            metric: Distance metric to use
            metric_params: Additional keyword arguments for the metric function
            n_jobs: Number of parallel jobs to run
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params or {}
        self.n_jobs = n_jobs
        
        # Training data
        self.X_train = None
        self.y_train = None
        self.is_classifier = None
        
        # Validate parameters
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        
        if weights not in ['uniform', 'distance']:
            raise ValueError("weights must be 'uniform' or 'distance'")
        
        if metric not in ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine']:
            raise ValueError(f"Unknown metric: {metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KNN model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Returns the instance itself
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if X.shape[0] < self.n_neighbors:
            warnings.warn(f"n_neighbors ({self.n_neighbors}) is larger than "
                         f"the number of samples ({X.shape[0]})")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Determine if this is a classification or regression problem
        if np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 0.1 * len(y):
            self.is_classifier = True
        else:
            self.is_classifier = False
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target for the provided data.
        
        Args:
            X: Test data of shape (n_samples, n_features)
        
        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, but KNN was fitted with "
                           f"{self.X_train.shape[1]} features")
        
        predictions = []
        
        for x in X:
            # Find k nearest neighbors
            distances = self._compute_distances(x, self.X_train)
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_distances = distances[k_nearest_indices]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # Make prediction based on neighbors
            if self.is_classifier:
                prediction = self._predict_classification(k_nearest_labels, k_nearest_distances)
            else:
                prediction = self._predict_regression(k_nearest_labels, k_nearest_distances)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities for classification.
        
        Args:
            X: Test data of shape (n_samples, n_features)
        
        Returns:
            Probability estimates of shape (n_samples, n_classes)
        """
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification")
        
        if self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        probabilities = []
        
        for x in X:
            # Find k nearest neighbors
            distances = self._compute_distances(x, self.X_train)
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_distances = distances[k_nearest_indices]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # Calculate class probabilities
            class_probs = np.zeros(n_classes)
            
            if self.weights == 'uniform':
                weights = np.ones(len(k_nearest_labels))
            else:  # distance weighting
                weights = 1 / (k_nearest_distances + 1e-8)  # Add small epsilon to avoid division by zero
            
            for i, class_label in enumerate(classes):
                mask = k_nearest_labels == class_label
                class_probs[i] = np.sum(weights[mask])
            
            # Normalize to get probabilities
            class_probs /= np.sum(class_probs)
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy (classification) or R² score (regression).
        
        Args:
            X: Test data of shape (n_samples, n_features)
            y: True values of shape (n_samples,)
        
        Returns:
            Score (accuracy for classification, R² for regression)
        """
        predictions = self.predict(X)
        
        if self.is_classifier:
            # Accuracy for classification
            return np.mean(predictions == y)
        else:
            # R² score for regression
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def kneighbors(self, X: Optional[np.ndarray] = None, n_neighbors: Optional[int] = None,
                   return_distance: bool = True):
        """
        Find the K-neighbors of a point.
        
        Args:
            X: Query points. If None, use training data
            n_neighbors: Number of neighbors to get. If None, use self.n_neighbors
            return_distance: Whether to return distances
        
        Returns:
            distances, indices (if return_distance=True) or just indices
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before calling kneighbors")
        
        if X is None:
            X = self.X_train
        else:
            X = np.asarray(X)
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        distances_list = []
        indices_list = []
        
        for x in X:
            distances = self._compute_distances(x, self.X_train)
            k_nearest_indices = np.argsort(distances)[:n_neighbors]
            k_nearest_distances = distances[k_nearest_indices]
            
            distances_list.append(k_nearest_distances)
            indices_list.append(k_nearest_indices)
        
        distances_array = np.array(distances_list)
        indices_array = np.array(indices_list)
        
        if return_distance:
            return distances_array, indices_array
        else:
            return indices_array
    
    def _compute_distances(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute distances between a point and a set of points.
        
        Args:
            x: Single point of shape (n_features,)
            X: Set of points of shape (n_samples, n_features)
        
        Returns:
            Distances of shape (n_samples,)
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((X - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X - x), axis=1)
        elif self.metric == 'minkowski':
            return np.sum(np.abs(X - x) ** self.p, axis=1) ** (1 / self.p)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(X - x), axis=1)
        elif self.metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(X, x)
            norms = np.linalg.norm(X, axis=1) * np.linalg.norm(x)
            cosine_sim = dot_product / (norms + 1e-8)
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _predict_classification(self, labels: np.ndarray, distances: np.ndarray):
        """
        Make classification prediction based on neighbor labels.
        
        Args:
            labels: Labels of k nearest neighbors
            distances: Distances to k nearest neighbors
        
        Returns:
            Predicted class label
        """
        if self.weights == 'uniform':
            # Simple majority vote
            return Counter(labels).most_common(1)[0][0]
        else:  # distance weighting
            # Weighted vote based on inverse distance
            weights = 1 / (distances + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Calculate weighted votes for each class
            unique_labels = np.unique(labels)
            weighted_votes = {}
            
            for label in unique_labels:
                mask = labels == label
                weighted_votes[label] = np.sum(weights[mask])
            
            # Return class with highest weighted vote
            return max(weighted_votes, key=weighted_votes.get)
    
    def _predict_regression(self, values: np.ndarray, distances: np.ndarray):
        """
        Make regression prediction based on neighbor values.
        
        Args:
            values: Values of k nearest neighbors
            distances: Distances to k nearest neighbors
        
        Returns:
            Predicted value
        """
        if self.weights == 'uniform':
            # Simple average
            return np.mean(values)
        else:  # distance weighting
            # Weighted average based on inverse distance
            weights = 1 / (distances + 1e-8)  # Add small epsilon to avoid division by zero
            return np.average(values, weights=weights)
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, return parameters for sub-estimators too
        
        Returns:
            Parameter names mapped to their values
        """
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'p': self.p,
            'metric': self.metric,
            'metric_params': self.metric_params,
            'n_jobs': self.n_jobs
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


class KNeighborsClassifier(KNN):
    """
    K-Nearest Neighbors classifier.
    
    This is a specialized version of KNN for classification tasks.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 algorithm: str = 'auto', leaf_size: int = 30,
                 p: int = 2, metric: str = 'minkowski',
                 metric_params: Optional[dict] = None, n_jobs: Optional[int] = None):
        """Initialize KNN classifier."""
        super().__init__(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
        self.is_classifier = True


class KNeighborsRegressor(KNN):
    """
    K-Nearest Neighbors regressor.
    
    This is a specialized version of KNN for regression tasks.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 algorithm: str = 'auto', leaf_size: int = 30,
                 p: int = 2, metric: str = 'minkowski',
                 metric_params: Optional[dict] = None, n_jobs: Optional[int] = None):
        """Initialize KNN regressor."""
        super().__init__(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
        self.is_classifier = False