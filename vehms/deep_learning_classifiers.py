# ============================================================
# Deep Learning Classifiers Module
# ============================================================
# Purpose: CNN, LSTM, and Hybrid classifiers for VEHMS
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
    LSTM, Bidirectional, Input, Concatenate, BatchNormalization,
    GlobalAveragePooling1D, Reshape, Layer, GRU
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.base import BaseEstimator, ClassifierMixin
from .config import RANDOM_SEED

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    1D-CNN Classifier for tabular data.
    
    Wraps a Keras CNN model in a sklearn-compatible interface for use
    with scikit-learn's cross-validation and stacking utilities.
    
    Architecture:
    - Input reshape to (features, 1) for Conv1D
    - Multiple Conv1D blocks with BatchNorm and Dropout
    - Global Average Pooling
    - Dense classification head
    
    Parameters:
    -----------
    n_features : int
        Number of input features (default: 9 for VEHMS)
    n_classes : int
        Number of output classes (default: 4 for VEHMS)
    filters : list
        Number of filters for each Conv1D layer
    kernel_size : int
        Kernel size for Conv1D layers
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for Adam optimizer
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_features=9, n_classes=4, filters=None,
                 kernel_size=3, dropout_rate=0.3, learning_rate=0.001,
                 epochs=100, batch_size=32, random_state=RANDOM_SEED,
                 verbose=0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.filters = filters if filters is not None else [64, 128, 256]
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.history = None
        self.classes_ = None

    def _build_model(self):
        """Build the CNN architecture."""
        tf.random.set_seed(self.random_state)
        
        model = Sequential([
            # Reshape input for Conv1D: (batch, features, 1)
            Reshape((self.n_features, 1), input_shape=(self.n_features,)),
            
            # First Conv Block
            Conv1D(self.filters[0], self.kernel_size, activation='relu', 
                   padding='same', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, padding='same'),
            Dropout(self.dropout_rate),
            
            # Second Conv Block
            Conv1D(self.filters[1], self.kernel_size, activation='relu', 
                   padding='same', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, padding='same'),
            Dropout(self.dropout_rate),
            
            # Third Conv Block
            Conv1D(self.filters[2], self.kernel_size, activation='relu', 
                   padding='same', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X, y):
        """
        Fit the CNN classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Fitted classifier
        """
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        # Update n_features based on input
        self.n_features = X.shape[1]
        
        self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, 
                         monitor='val_loss', verbose=self.verbose),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, 
                             verbose=self.verbose)
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X, verbose=0)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    LSTM Classifier for tabular data.
    
    Treats features as a sequence to capture dependencies between features.
    Supports both unidirectional and bidirectional LSTM.
    
    Architecture:
    - Input reshape to (features, 1) for LSTM
    - Stacked LSTM layers with BatchNorm
    - Dense classification head
    
    Parameters:
    -----------
    n_features : int
        Number of input features (default: 9 for VEHMS)
    n_classes : int
        Number of output classes (default: 4 for VEHMS)
    lstm_units : list
        Number of units for each LSTM layer
    dropout_rate : float
        Dropout rate for regularization
    recurrent_dropout : float
        Dropout rate for recurrent connections
    learning_rate : float
        Learning rate for Adam optimizer
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    bidirectional : bool
        Whether to use bidirectional LSTM
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_features=9, n_classes=4, lstm_units=None,
                 dropout_rate=0.3, recurrent_dropout=0.2, learning_rate=0.001,
                 epochs=100, batch_size=32, bidirectional=False,
                 random_state=RANDOM_SEED, verbose=0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lstm_units = lstm_units if lstm_units is not None else [64, 32]
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.history = None
        self.classes_ = None
    
    def _build_model(self):
        """Build the LSTM architecture."""
        tf.random.set_seed(self.random_state)
        
        model = Sequential()
        
        # Reshape input for LSTM: (batch, timesteps, features)
        model.add(Reshape((self.n_features, 1), input_shape=(self.n_features,)))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            
            lstm_layer = LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l2(0.001)
            )
            
            if self.bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            
            model.add(lstm_layer)
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(self.n_classes, activation='softmax'))
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X, y):
        """
        Fit the LSTM classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Fitted classifier
        """
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        # Update n_features based on input
        self.n_features = X.shape[1]
        
        self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True,
                         monitor='val_loss', verbose=self.verbose),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6,
                             verbose=self.verbose)
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X, verbose=0)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout': self.recurrent_dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'bidirectional': self.bidirectional,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CNNLSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid CNN-LSTM Classifier.
    
    Combines CNN's local feature extraction with LSTM's sequential modeling.
    CNN layers extract local patterns, LSTM captures dependencies.
    
    Architecture:
    - Input reshape to (features, 1)
    - Conv1D blocks for feature extraction
    - LSTM layers for sequential modeling
    - Dense classification head
    
    Parameters:
    -----------
    n_features : int
        Number of input features (default: 9 for VEHMS)
    n_classes : int
        Number of output classes (default: 4 for VEHMS)
    cnn_filters : list
        Number of filters for each Conv1D layer
    lstm_units : list
        Number of units for each LSTM layer
    kernel_size : int
        Kernel size for Conv1D layers
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for Adam optimizer
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    bidirectional : bool
        Whether to use bidirectional LSTM
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_features=9, n_classes=4, cnn_filters=None,
                 lstm_units=None, kernel_size=3, dropout_rate=0.3,
                 learning_rate=0.001, epochs=100, batch_size=32,
                 bidirectional=False, random_state=RANDOM_SEED, verbose=0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.cnn_filters = cnn_filters if cnn_filters is not None else [64, 128]
        self.lstm_units = lstm_units if lstm_units is not None else [64]
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.history = None
        self.classes_ = None
    
    def _build_model(self):
        """Build the CNN-LSTM hybrid architecture."""
        tf.random.set_seed(self.random_state)
        
        inputs = Input(shape=(self.n_features,))
        
        # Reshape for Conv1D
        x = Reshape((self.n_features, 1))(inputs)
        
        # CNN feature extraction
        for filters in self.cnn_filters:
            x = Conv1D(filters, self.kernel_size, activation='relu', 
                      padding='same', kernel_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # LSTM sequential modeling
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            
            lstm_layer = LSTM(units, return_sequences=return_sequences,
                             dropout=self.dropout_rate,
                             kernel_regularizer=l2(0.001))
            
            if self.bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            
            x = lstm_layer(x)
            x = BatchNormalization()(x)
        
        # Dense classification head
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X, y):
        """
        Fit the CNN-LSTM classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Fitted classifier
        """
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        # Update n_features based on input
        self.n_features = X.shape[1]
        
        self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True,
                         monitor='val_loss', verbose=self.verbose),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6,
                             verbose=self.verbose)
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X, verbose=0)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'cnn_filters': self.cnn_filters,
            'lstm_units': self.lstm_units,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'bidirectional': self.bidirectional,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class GRUClassifier(BaseEstimator, ClassifierMixin):
    """
    GRU Classifier for tabular data.
    
    Similar to LSTM but with simpler architecture (fewer parameters).
    Often trains faster while achieving comparable performance.
    
    Parameters:
    -----------
    n_features : int
        Number of input features (default: 9 for VEHMS)
    n_classes : int
        Number of output classes (default: 4 for VEHMS)
    gru_units : list
        Number of units for each GRU layer
    dropout_rate : float
        Dropout rate for regularization
    recurrent_dropout : float
        Dropout rate for recurrent connections
    learning_rate : float
        Learning rate for Adam optimizer
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    bidirectional : bool
        Whether to use bidirectional GRU
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_features=9, n_classes=4, gru_units=None,
                 dropout_rate=0.3, recurrent_dropout=0.2, learning_rate=0.001,
                 epochs=100, batch_size=32, bidirectional=False,
                 random_state=RANDOM_SEED, verbose=0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.gru_units = gru_units if gru_units is not None else [64, 32]
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.history = None
        self.classes_ = None
    
    def _build_model(self):
        """Build the GRU architecture."""
        tf.random.set_seed(self.random_state)
        
        model = Sequential()
        
        # Reshape input for GRU: (batch, timesteps, features)
        model.add(Reshape((self.n_features, 1), input_shape=(self.n_features,)))
        
        # GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = (i < len(self.gru_units) - 1)
            
            gru_layer = GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l2(0.001)
            )
            
            if self.bidirectional:
                gru_layer = Bidirectional(gru_layer)
            
            model.add(gru_layer)
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(self.n_classes, activation='softmax'))
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X, y):
        """Fit the GRU classifier."""
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        
        self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True,
                         monitor='val_loss', verbose=self.verbose),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6,
                             verbose=self.verbose)
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X, verbose=0)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'gru_units': self.gru_units,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout': self.recurrent_dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'bidirectional': self.bidirectional,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class AttentionLSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    LSTM Classifier with Attention Mechanism.
    
    Adds attention layer to focus on important features/timesteps.
    Provides interpretability through attention weights.
    
    Architecture:
    - Input reshape to (features, 1)
    - LSTM layers with return_sequences=True
    - Custom attention mechanism
    - Dense classification head
    
    Parameters:
    -----------
    n_features : int
        Number of input features (default: 9 for VEHMS)
    n_classes : int
        Number of output classes (default: 4 for VEHMS)
    lstm_units : list
        Number of units for each LSTM layer
    attention_units : int
        Number of units in attention layer
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for Adam optimizer
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_features=9, n_classes=4, lstm_units=None,
                 attention_units=64, dropout_rate=0.3, learning_rate=0.001,
                 epochs=100, batch_size=32, random_state=RANDOM_SEED, verbose=0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lstm_units = lstm_units if lstm_units is not None else [64, 64]
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.history = None
        self.classes_ = None
        self.attention_model = None
    
    def _build_model(self):
        """Build the Attention-LSTM architecture."""
        tf.random.set_seed(self.random_state)
        
        inputs = Input(shape=(self.n_features,))
        
        # Reshape for LSTM
        x = Reshape((self.n_features, 1))(inputs)
        
        # LSTM layers (all return sequences for attention)
        for units in self.lstm_units:
            x = LSTM(units, return_sequences=True, dropout=self.dropout_rate,
                    kernel_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
        
        # Attention mechanism
        # Compute attention scores
        attention_scores = Dense(self.attention_units, activation='tanh',
                                kernel_regularizer=l2(0.001))(x)
        attention_scores = Dense(1, activation='softmax')(attention_scores)
        
        # Apply attention weights
        context = tf.keras.layers.Multiply()([x, attention_scores])
        context = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        
        # Dense classification head
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(context)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create attention model for interpretability
        self.attention_model = Model(inputs, attention_scores)
        
        return model
    
    def fit(self, X, y):
        """Fit the Attention-LSTM classifier."""
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        
        self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True,
                         monitor='val_loss', verbose=self.verbose),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6,
                             verbose=self.verbose)
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X, verbose=0)
    
    def get_attention_weights(self, X):
        """Get attention weights for interpretability."""
        if self.attention_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.attention_model.predict(X, verbose=0)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'lstm_units': self.lstm_units,
            'attention_units': self.attention_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
