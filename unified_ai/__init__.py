"""
Unified AI Framework - A comprehensive deep learning, machine learning, and NLP library.

This package provides:
- Core deep learning components (tensors, layers, models)
- Traditional machine learning algorithms
- State-of-the-art deep learning architectures (GAN, YOLO, Transformer)
- Complete NLP pipeline (tokenization, embeddings, language models)
- Utilities for training, evaluation, and visualization
"""

__version__ = "0.1.0"
__author__ = "Unified AI Team"
__email__ = "team@unified-ai.org"

# Core imports
from unified_ai.core.tensor import Tensor
from unified_ai.core.model import Model
from unified_ai.core.layers import Dense, Conv2D, BatchNorm, Dropout, Embedding, MultiHeadAttention
from unified_ai.core.optimizer import SGD, Adam, RMSProp

# ML algorithms
from unified_ai.ml.knn import KNN
from unified_ai.ml.svm import SVM
from unified_ai.ml.decision_tree import DecisionTree
from unified_ai.ml.naive_bayes import GaussianNB, MultinomialNB
from unified_ai.ml.linear_regression import LinearRegression, LogisticRegression

# Deep learning models
from unified_ai.models.gan import DCGAN
from unified_ai.models.yolo import TinyYOLO
from unified_ai.models.transformer import Transformer

# NLP components
from unified_ai.nlp.tokenizer import BPETokenizer, WordPieceTokenizer
from unified_ai.nlp.embedding import Word2Vec, GloVe
from unified_ai.nlp.language_model import GPTLanguageModel

# Utilities
from unified_ai.utils.losses import MSELoss, CrossEntropyLoss, BCELoss
from unified_ai.utils.metrics import accuracy_score, precision_score, recall_score, f1_score
from unified_ai.callbacks.early_stopping import EarlyStopping
from unified_ai.callbacks.model_checkpoint import ModelCheckpoint

__all__ = [
    # Core
    'Tensor', 'Model', 'Dense', 'Conv2D', 'BatchNorm', 'Dropout', 'Embedding', 'MultiHeadAttention',
    'SGD', 'Adam', 'RMSProp',
    
    # ML
    'KNN', 'SVM', 'DecisionTree', 'GaussianNB', 'MultinomialNB', 
    'LinearRegression', 'LogisticRegression',
    
    # Deep Learning
    'DCGAN', 'TinyYOLO', 'Transformer',
    
    # NLP
    'BPETokenizer', 'WordPieceTokenizer', 'Word2Vec', 'GloVe', 'GPTLanguageModel',
    
    # Utils
    'MSELoss', 'CrossEntropyLoss', 'BCELoss',
    'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'EarlyStopping', 'ModelCheckpoint'
]