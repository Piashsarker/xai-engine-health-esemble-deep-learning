# ============================================================
# Hyperparameter Tuning Module for Deep Learning Classifiers
# ============================================================
# Purpose: Systematic hyperparameter optimization for VEHMS
# ============================================================

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import itertools
from typing import Dict, List, Tuple, Any
import time
from .config import RANDOM_SEED

# Set random seeds
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DeepLearningHyperparameterTuner:
    """
    Hyperparameter tuner for deep learning classifiers.
    
    Supports grid search and random search for CNN, LSTM, GRU,
    CNN-LSTM, BiLSTM, and Attention-LSTM classifiers.
    
    Parameters:
    -----------
    classifier_type : str
        Type of classifier: 'cnn', 'lstm', 'gru', 'cnn_lstm', 
        'bilstm', 'attention_lstm'
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric ('accuracy', 'f1_weighted')
    random_state : int
        Random seed for reproducibility
    verbose : int
        Verbosity level
    """
    
    def __init__(self, classifier_type='cnn', cv=3, scoring='accuracy',
                 random_state=RANDOM_SEED, verbose=1):
        self.classifier_type = classifier_type
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = []
    
    def _get_classifier_class(self):
        """Get the classifier class based on type."""
        from .deep_learning_classifiers import (
            CNNClassifier, LSTMClassifier, CNNLSTMClassifier,
            GRUClassifier, AttentionLSTMClassifier
        )
        
        classifiers = {
            'cnn': CNNClassifier,
            'lstm': LSTMClassifier,
            'gru': GRUClassifier,
            'cnn_lstm': CNNLSTMClassifier,
            'bilstm': lambda **kwargs: LSTMClassifier(bidirectional=True, **kwargs),
            'attention_lstm': AttentionLSTMClassifier
        }
        return classifiers.get(self.classifier_type)
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid based on classifier type."""
        
        # Common parameters for all models
        common_params = {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'dropout_rate': [0.2, 0.3, 0.4],
            'batch_size': [32, 64],
            'epochs': [150]  # Use early stopping
        }
        
        if self.classifier_type == 'cnn':
            return {
                **common_params,
                'filters': [[32, 64, 128], [64, 128, 256], [128, 256, 512]],
                'kernel_size': [3, 5]
            }
        
        elif self.classifier_type in ['lstm', 'bilstm']:
            return {
                **common_params,
                'lstm_units': [[128, 64], [64, 32], [128, 64, 32]],
                'recurrent_dropout': [0.1, 0.2, 0.3]
            }
        
        elif self.classifier_type == 'gru':
            return {
                **common_params,
                'gru_units': [[128, 64], [64, 32], [128, 64, 32]],
                'recurrent_dropout': [0.1, 0.2, 0.3]
            }
        
        elif self.classifier_type == 'cnn_lstm':
            return {
                **common_params,
                'cnn_filters': [[64, 128], [128, 256]],
                'lstm_units': [[64], [128], [64, 32]],
                'kernel_size': [3, 5]
            }
        
        elif self.classifier_type == 'attention_lstm':
            return {
                **common_params,
                'lstm_units': [[128, 64], [64, 64], [128, 128]],
                'attention_units': [32, 64, 128]
            }
        
        return common_params
    
    def _cross_validate(self, clf_class, params: Dict, X, y) -> float:
        """Perform cross-validation for a parameter set."""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, 
                              random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train classifier
            clf = clf_class(random_state=self.random_state, verbose=0, **params)
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
            
            # Clear session to free memory
            tf.keras.backend.clear_session()
        
        return np.mean(scores), np.std(scores)
    
    def grid_search(self, X, y, param_grid: Dict[str, List] = None) -> Dict:
        """
        Perform grid search over parameter combinations.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        param_grid : dict
            Parameter grid (uses default if None)
            
        Returns:
        --------
        dict : Best parameters and results
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        clf_class = self._get_classifier_class()
        if clf_class is None:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"GRID SEARCH: {self.classifier_type.upper()}")
            print(f"{'='*60}")
            print(f"Total combinations: {len(param_combinations)}")
            print(f"CV Folds: {self.cv}")
        
        self.cv_results_ = []
        self.best_score_ = -np.inf
        self.best_params_ = None
        
        for i, values in enumerate(param_combinations):
            params = dict(zip(param_names, values))
            
            if self.verbose >= 1:
                print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
            
            start_time = time.time()
            mean_score, std_score = self._cross_validate(clf_class, params, X, y)
            elapsed = time.time() - start_time
            
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'time': elapsed
            }
            self.cv_results_.append(result)
            
            if self.verbose >= 1:
                print(f"   Score: {mean_score:.4f} (+/- {std_score:.4f}) [{elapsed:.1f}s]")
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        
        # Train best model on full data
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"BEST PARAMETERS: {self.best_params_}")
            print(f"BEST CV SCORE: {self.best_score_:.4f}")
            print(f"{'='*60}")
        
        self.best_estimator_ = clf_class(
            random_state=self.random_state, 
            verbose=0, 
            **self.best_params_
        )
        self.best_estimator_.fit(X, y)
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.cv_results_
        }
    
    def random_search(self, X, y, param_distributions: Dict = None, 
                      n_iter: int = 20) -> Dict:
        """
        Perform random search over parameter distributions.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        param_distributions : dict
            Parameter distributions (uses default grid if None)
        n_iter : int
            Number of random combinations to try
            
        Returns:
        --------
        dict : Best parameters and results
        """
        if param_distributions is None:
            param_distributions = self._get_default_param_grid()
        
        clf_class = self._get_classifier_class()
        if clf_class is None:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"RANDOM SEARCH: {self.classifier_type.upper()}")
            print(f"{'='*60}")
            print(f"Iterations: {n_iter}")
            print(f"CV Folds: {self.cv}")
        
        self.cv_results_ = []
        self.best_score_ = -np.inf
        self.best_params_ = None
        
        np.random.seed(self.random_state)
        
        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for name, values in param_distributions.items():
                params[name] = np.random.choice(values) if isinstance(values[0], (int, float, str)) else values[np.random.randint(len(values))]
            
            if self.verbose >= 1:
                print(f"\n[{i+1}/{n_iter}] Testing: {params}")
            
            start_time = time.time()
            mean_score, std_score = self._cross_validate(clf_class, params, X, y)
            elapsed = time.time() - start_time
            
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'time': elapsed
            }
            self.cv_results_.append(result)
            
            if self.verbose >= 1:
                print(f"   Score: {mean_score:.4f} (+/- {std_score:.4f}) [{elapsed:.1f}s]")
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"BEST PARAMETERS: {self.best_params_}")
            print(f"BEST CV SCORE: {self.best_score_:.4f}")
            print(f"{'='*60}")
        
        self.best_estimator_ = clf_class(
            random_state=self.random_state, 
            verbose=0, 
            **self.best_params_
        )
        self.best_estimator_.fit(X, y)
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.cv_results_
        }


class OptimizedHyperparameters:
    """
    Pre-defined optimized hyperparameters for VEHMS deep learning models.
    
    These parameters have been tuned for the VEHMS dataset and provide
    a good starting point for achieving high accuracy.
    """
    
    # Optimized parameters for each model type
    CNN = {
        'filters': [128, 256, 512],
        'kernel_size': 3,
        'dropout_rate': 0.25,
        'learning_rate': 0.0005,
        'epochs': 150,
        'batch_size': 32
    }
    
    LSTM = {
        'lstm_units': [128, 64],
        'dropout_rate': 0.25,
        'recurrent_dropout': 0.15,
        'learning_rate': 0.0005,
        'epochs': 150,
        'batch_size': 32
    }
    
    BiLSTM = {
        'lstm_units': [64, 32],
        'dropout_rate': 0.25,
        'recurrent_dropout': 0.15,
        'learning_rate': 0.0005,
        'bidirectional': True,
        'epochs': 150,
        'batch_size': 32
    }
    
    GRU = {
        'gru_units': [128, 64],
        'dropout_rate': 0.25,
        'recurrent_dropout': 0.15,
        'learning_rate': 0.0005,
        'epochs': 150,
        'batch_size': 32
    }
    
    CNN_LSTM = {
        'cnn_filters': [128, 256],
        'lstm_units': [64],
        'kernel_size': 3,
        'dropout_rate': 0.25,
        'learning_rate': 0.0005,
        'epochs': 150,
        'batch_size': 32
    }
    
    ATTENTION_LSTM = {
        'lstm_units': [128, 64],
        'attention_units': 64,
        'dropout_rate': 0.25,
        'learning_rate': 0.0005,
        'epochs': 150,
        'batch_size': 32
    }
    
    @classmethod
    def get_params(cls, model_type: str) -> Dict:
        """Get optimized parameters for a model type."""
        params_map = {
            'cnn': cls.CNN,
            'lstm': cls.LSTM,
            'bilstm': cls.BiLSTM,
            'gru': cls.GRU,
            'cnn_lstm': cls.CNN_LSTM,
            'attention_lstm': cls.ATTENTION_LSTM
        }
        return params_map.get(model_type.lower(), {})


def tune_all_models(X_train, y_train, cv=3, n_iter=10, verbose=1):
    """
    Tune all deep learning models and return best parameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int
        Cross-validation folds
    n_iter : int
        Number of random search iterations per model
    verbose : int
        Verbosity level
        
    Returns:
    --------
    dict : Best parameters for each model type
    """
    model_types = ['cnn', 'lstm', 'gru', 'cnn_lstm', 'bilstm', 'attention_lstm']
    results = {}
    
    for model_type in model_types:
        if verbose >= 1:
            print(f"\n{'#'*60}")
            print(f"# TUNING: {model_type.upper()}")
            print(f"{'#'*60}")
        
        tuner = DeepLearningHyperparameterTuner(
            classifier_type=model_type,
            cv=cv,
            verbose=verbose
        )
        
        result = tuner.random_search(X_train, y_train, n_iter=n_iter)
        results[model_type] = {
            'best_params': result['best_params'],
            'best_score': result['best_score'],
            'best_estimator': tuner.best_estimator_
        }
    
    return results
