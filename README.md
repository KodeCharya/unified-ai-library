# 🧠 Unified AI Framework

A powerful and extensible deep learning + machine learning + NLP framework built in Python. Designed to be educational, modular, and production-ready.

## 🚀 Features

### Core Framework
- **Tensor abstraction** using NumPy with automatic differentiation support
- **Layer classes**: Dense, Conv2D, BatchNorm, Dropout, Embedding, Attention, etc.
- **Model class** (Keras-like): supports `compile()`, `fit()`, `evaluate()`, `predict()`
- **Optimizers**: SGD, Adam, RMSProp with momentum and weight decay
- **Callbacks**: EarlyStopping, ModelCheckpoint, LearningRateScheduler
- **Loss functions**: CrossEntropy, MSE, BCE, Focal Loss
- **Metrics**: Accuracy, Precision, Recall, F1, BLEU, IoU

### Machine Learning (from scratch)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree with pruning
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Linear/Logistic Regression with regularization

### Deep Learning Models
- **GAN**: Deep Convolutional GAN (DCGAN) with Wasserstein loss
- **YOLO**: Tiny YOLO for object detection with NMS and anchor boxes
- **Transformer**: GPT-like transformer with multi-head attention

### NLP Pipeline
- **Tokenizer**: Byte Pair Encoding (BPE) and WordPiece
- **Embeddings**: Word2Vec, GloVe with pre-trained support
- **Language Model**: Transformer-based causal language modeling
- **Preprocessing**: Text cleaning, stemming, lemmatization

## 📦 Installation

```bash
pip install unified_ai
```

Or install from source:

```bash
git clone https://github.com/unified-ai/unified_ai.git
cd unified_ai
pip install -e .
```

## 🏃‍♂️ Quick Start

### Deep Learning Example

```python
from unified_ai.core import Model, Dense, Dropout
from unified_ai.utils.losses import CrossEntropyLoss
from unified_ai.core.optimizer import Adam

# Create a simple neural network
model = Model([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CrossEntropyLoss(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(X_train, y_train, 
                   validation_data=(X_val, y_val),
                   epochs=50, batch_size=32)
```

### Machine Learning Example

```python
from unified_ai.ml import SVM, DecisionTree
from unified_ai.utils.metrics import accuracy_score

# Support Vector Machine
svm = SVM(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred)}")

# Decision Tree
dt = DecisionTree(max_depth=10, min_samples_split=5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred)}")
```

### NLP Example

```python
from unified_ai.nlp import BPETokenizer, Word2Vec
from unified_ai.models import Transformer

# Tokenization
tokenizer = BPETokenizer(vocab_size=10000)
tokenizer.fit(corpus)
tokens = tokenizer.encode("Hello, world!")

# Word embeddings
w2v = Word2Vec(vector_size=300, window=5, min_count=1)
w2v.fit(sentences)
embedding = w2v.get_vector("hello")

# Transformer language model
model = Transformer(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=1024
)
```

## 🏗️ Architecture

```
unified_ai/
├── core/           # Core framework components
├── callbacks/      # Training callbacks
├── ml/            # Traditional ML algorithms
├── models/        # Deep learning architectures
├── nlp/           # NLP components
├── utils/         # Utilities and helpers
├── examples/      # Example scripts
└── tests/         # Unit tests
```

## 🧪 Examples

Check out the `examples/` directory for complete training scripts:

- `train_gan.py` - Train a DCGAN on MNIST
- `train_yolo.py` - Object detection with YOLO
- `train_transformer.py` - Language modeling with Transformer
- `train_ml_models.py` - Traditional ML algorithms comparison
- `nlp_pipeline_example.py` - Complete NLP pipeline

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by TensorFlow/Keras, PyTorch, and scikit-learn
- Built for educational purposes and research
- Community-driven development