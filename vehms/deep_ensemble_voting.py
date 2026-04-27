# ============================================================
# Deep Ensemble Voting Module
# ============================================================
# Purpose: Advanced ensemble voting for deep learning classifiers
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import time
from .config import RANDOM_SEED

# Set random seeds
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DeepEnsembleVoting(BaseEstimator, ClassifierMixin):
    """
    Advanced Ensemble Voting for Deep Learning Classifiers.
    
    Combines multiple deep learning models using various voting strategies
    to achieve better accuracy than individual models.
    
    Voting Strategies:
    - 'hard': Majority voting on predicted classes
    - 'soft': Average of predicted probabilities
    - 'weighted_soft': Weighted average based on validation accuracy
    - 'weighted_hard': Weighted majority voting
    - 'stacked': Use a meta-learner on predictions
    
    Parameters:
    -----------
    estimators : list of tuples
        List of (name, estimator) tuples
    voting : str
        Voting strategy: 'hard', 'soft', 'weighted_soft', 'weighted_hard', 'stacked'
    weights : list or None
        Weights for each estimator (auto-calculated if None for weighted strategies)
    meta_learner : estimator or None
        Meta-learner for stacked voting (default: LogisticRegression)
    random_state : int
        Random seed for reproducibility
    verbose : int
        Verbosity level
    """
    
    def __init__(self, estimators, voting='weighted_soft', weights=None,
                 meta_learner=None, random_state=RANDOM_SEED, verbose=1):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.meta_learner = meta_learner
        self.random_state = random_state
        self.verbose = verbose
        self.fitted_estimators_ = []
        self.classes_ = None
        self.n_classes_ = None
        self.weights_ = None
        self.training_time_ = None
        self.individual_accuracies_ = {}
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit all estimators in the ensemble.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        X_val : array-like, optional
            Validation features (for weight calculation)
        y_val : array-like, optional
            Validation labels (for weight calculation)
            
        Returns:
        --------
        self : object
            Fitted ensemble
        """
        start_time = time.time()
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.fitted_estimators_ = []
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"DEEP ENSEMBLE VOTING - {self.voting.upper()}")
            print(f"{'='*60}")
            print(f"Number of estimators: {len(self.estimators)}")
        
        # Fit each estimator
        for name, estimator in self.estimators:
            if self.verbose >= 1:
                print(f"\n[...] Training: {name}")
            
            estimator.fit(X, y)
            self.fitted_estimators_.append((name, estimator))
            
            # Calculate training accuracy
            train_acc = accuracy_score(y, estimator.predict(X))
            if self.verbose >= 1:
                print(f"   [OK] {name} - Train Accuracy: {train_acc:.4f}")
        
        # Calculate weights if needed
        if self.voting in ['weighted_soft', 'weighted_hard'] and self.weights is None:
            if X_val is not None and y_val is not None:
                self._calculate_weights(X_val, y_val)
            else:
                # Use training accuracy as proxy
                self._calculate_weights(X, y)
        elif self.weights is not None:
            self.weights_ = np.array(self.weights)
            self.weights_ = self.weights_ / self.weights_.sum()
        
        # Fit meta-learner for stacked voting
        if self.voting == 'stacked':
            self._fit_meta_learner(X, y)
        
        self.training_time_ = time.time() - start_time
        
        if self.verbose >= 1:
            print(f"\n[OK] Ensemble trained in {self.training_time_:.2f} seconds")
            if self.weights_ is not None:
                print(f"\nEstimator Weights:")
                for (name, _), w in zip(self.fitted_estimators_, self.weights_):
                    print(f"   {name}: {w:.4f}")
            print(f"{'='*60}")
        
        return self
    
    def _calculate_weights(self, X, y):
        """Calculate weights based on validation accuracy."""
        accuracies = []
        
        for name, estimator in self.fitted_estimators_:
            y_pred = estimator.predict(X)
            acc = accuracy_score(y, y_pred)
            accuracies.append(acc)
            self.individual_accuracies_[name] = acc
        
        # Convert to weights (higher accuracy = higher weight)
        accuracies = np.array(accuracies)
        
        # Use softmax-like weighting to emphasize better models
        # Raise to power to increase differentiation
        weights = accuracies ** 2
        self.weights_ = weights / weights.sum()
    
    def _fit_meta_learner(self, X, y):
        """Fit meta-learner for stacked voting."""
        from sklearn.linear_model import LogisticRegression
        
        if self.meta_learner is None:
            self.meta_learner = LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state
            )
        
        # Generate meta-features
        meta_features = self._get_meta_features(X)
        self.meta_learner.fit(meta_features, y)
    
    def _get_meta_features(self, X):
        """Generate meta-features from all estimators."""
        meta_features = []
        
        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
                meta_features.append(proba)
            else:
                pred = estimator.predict(X)
                one_hot = np.eye(self.n_classes_)[pred]
                meta_features.append(one_hot)
        
        return np.hstack(meta_features)
    
    def predict(self, X):
        """
        Predict class labels using the ensemble.
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array : Predicted class labels
        """
        if self.voting == 'hard':
            return self._predict_hard(X)
        elif self.voting == 'soft':
            return self._predict_soft(X)
        elif self.voting == 'weighted_soft':
            return self._predict_weighted_soft(X)
        elif self.voting == 'weighted_hard':
            return self._predict_weighted_hard(X)
        elif self.voting == 'stacked':
            return self._predict_stacked(X)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")
    
    def _predict_hard(self, X):
        """Hard voting - majority vote."""
        predictions = np.array([
            estimator.predict(X) for _, estimator in self.fitted_estimators_
        ])
        # Majority vote
        return np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_).argmax(),
            axis=0, arr=predictions
        )
    
    def _predict_soft(self, X):
        """Soft voting - average probabilities."""
        probas = self._get_average_probas(X)
        return np.argmax(probas, axis=1)
    
    def _predict_weighted_soft(self, X):
        """Weighted soft voting."""
        probas = self._get_weighted_average_probas(X)
        return np.argmax(probas, axis=1)
    
    def _predict_weighted_hard(self, X):
        """Weighted hard voting."""
        predictions = np.array([
            estimator.predict(X) for _, estimator in self.fitted_estimators_
        ])
        
        # Weighted vote
        weighted_votes = np.zeros((X.shape[0], self.n_classes_))
        for i, pred in enumerate(predictions):
            for j, p in enumerate(pred):
                weighted_votes[j, p] += self.weights_[i]
        
        return np.argmax(weighted_votes, axis=1)
    
    def _predict_stacked(self, X):
        """Stacked prediction using meta-learner."""
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features)
    
    def _get_average_probas(self, X):
        """Get average probabilities from all estimators."""
        probas = []
        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, 'predict_proba'):
                probas.append(estimator.predict_proba(X))
            else:
                pred = estimator.predict(X)
                probas.append(np.eye(self.n_classes_)[pred])
        
        return np.mean(probas, axis=0)
    
    def _get_weighted_average_probas(self, X):
        """Get weighted average probabilities."""
        probas = np.zeros((X.shape[0], self.n_classes_))
        
        for i, (name, estimator) in enumerate(self.fitted_estimators_):
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X)
                proba = np.eye(self.n_classes_)[pred]
            
            probas += self.weights_[i] * proba
        
        return probas
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array : Predicted probabilities
        """
        if self.voting in ['soft', 'hard']:
            return self._get_average_probas(X)
        elif self.voting in ['weighted_soft', 'weighted_hard']:
            return self._get_weighted_average_probas(X)
        elif self.voting == 'stacked':
            meta_features = self._get_meta_features(X)
            if hasattr(self.meta_learner, 'predict_proba'):
                return self.meta_learner.predict_proba(meta_features)
            else:
                pred = self.meta_learner.predict(meta_features)
                return np.eye(self.n_classes_)[pred]
        else:
            return self._get_average_probas(X)
    
    def score(self, X, y):
        """Return accuracy score."""
        return accuracy_score(y, self.predict(X))
    
    def get_individual_predictions(self, X):
        """Get predictions from each individual estimator."""
        return {
            name: estimator.predict(X) 
            for name, estimator in self.fitted_estimators_
        }
    
    def evaluate(self, X, y, label_encoder=None):
        """
        Comprehensive evaluation of the ensemble.
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels
        label_encoder : LabelEncoder, optional
            For decoding labels
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='weighted'),
            'Recall': recall_score(y, y_pred, average='weighted'),
            'F1': f1_score(y, y_pred, average='weighted'),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred)
        }
        
        # AUC (multi-class)
        try:
            metrics['AUC'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = None
        
        return metrics
    
    def compare_with_individuals(self, X, y):
        """
        Compare ensemble performance with individual models.
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels
            
        Returns:
        --------
        DataFrame : Comparison results
        """
        results = []
        
        # Individual model results
        for name, estimator in self.fitted_estimators_:
            y_pred = estimator.predict(X)
            y_proba = estimator.predict_proba(X) if hasattr(estimator, 'predict_proba') else None
            
            result = {
                'Model': name,
                'Accuracy': accuracy_score(y, y_pred),
                'Precision': precision_score(y, y_pred, average='weighted'),
                'F1': f1_score(y, y_pred, average='weighted'),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred))
            }
            
            if y_proba is not None:
                try:
                    result['AUC'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                except:
                    result['AUC'] = None
            
            results.append(result)
        
        # Ensemble result
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        ensemble_result = {
            'Model': f'ENSEMBLE ({self.voting})',
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='weighted'),
            'F1': f1_score(y, y_pred, average='weighted'),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        try:
            ensemble_result['AUC'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        except:
            ensemble_result['AUC'] = None
        
        results.append(ensemble_result)
        
        return pd.DataFrame(results)


class AdaptiveDeepEnsemble(DeepEnsembleVoting):
    """
    Adaptive Deep Ensemble that automatically selects the best voting strategy.
    
    Tests multiple voting strategies and selects the one with best
    validation performance.
    
    Parameters:
    -----------
    estimators : list of tuples
        List of (name, estimator) tuples
    strategies : list
        Voting strategies to test
    random_state : int
        Random seed
    verbose : int
        Verbosity level
    """
    
    def __init__(self, estimators, strategies=None, random_state=RANDOM_SEED, verbose=1):
        super().__init__(estimators, voting='weighted_soft', random_state=random_state, verbose=verbose)
        self.strategies = strategies or ['hard', 'soft', 'weighted_soft', 'weighted_hard', 'stacked']
        self.best_strategy_ = None
        self.strategy_scores_ = {}
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit ensemble and select best voting strategy.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        X_val : array-like
            Validation features
        y_val : array-like
            Validation labels
            
        Returns:
        --------
        self : object
            Fitted ensemble
        """
        # First, fit all base estimators
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.fitted_estimators_ = []
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("ADAPTIVE DEEP ENSEMBLE")
            print(f"{'='*60}")
        
        # Fit estimators
        for name, estimator in self.estimators:
            if self.verbose >= 1:
                print(f"[...] Training: {name}")
            estimator.fit(X, y)
            self.fitted_estimators_.append((name, estimator))
        
        # Calculate weights for weighted strategies
        if X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)
        else:
            self._calculate_weights(X, y)
        
        # Test each strategy
        test_X = X_val if X_val is not None else X
        test_y = y_val if y_val is not None else y
        
        if self.verbose >= 1:
            print(f"\n[...] Testing voting strategies...")
        
        for strategy in self.strategies:
            self.voting = strategy
            
            if strategy == 'stacked':
                self._fit_meta_learner(X, y)
            
            y_pred = self.predict(test_X)
            score = accuracy_score(test_y, y_pred)
            self.strategy_scores_[strategy] = score
            
            if self.verbose >= 1:
                print(f"   {strategy}: {score:.4f}")
        
        # Select best strategy
        self.best_strategy_ = max(self.strategy_scores_, key=self.strategy_scores_.get)
        self.voting = self.best_strategy_
        
        if self.voting == 'stacked':
            self._fit_meta_learner(X, y)
        
        if self.verbose >= 1:
            print(f"\n[OK] Best Strategy: {self.best_strategy_} ({self.strategy_scores_[self.best_strategy_]:.4f})")
            print(f"{'='*60}")
        
        return self


def create_optimized_ensemble(X_train, y_train, X_val, y_val, 
                              include_models=None, voting='weighted_soft',
                              use_tuned_params=True, verbose=1):
    """
    Create an optimized deep learning ensemble.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    include_models : list, optional
        Models to include (default: best performing models)
    voting : str
        Voting strategy
    use_tuned_params : bool
        Whether to use pre-tuned hyperparameters
    verbose : int
        Verbosity level
        
    Returns:
    --------
    DeepEnsembleVoting : Fitted ensemble
    """
    from .deep_learning_classifiers import (
        CNNClassifier, LSTMClassifier, CNNLSTMClassifier,
        GRUClassifier, AttentionLSTMClassifier
    )
    from .hyperparameter_tuner import OptimizedHyperparameters
    
    if include_models is None:
        # Default: include best performing models
        include_models = ['cnn', 'cnn_lstm', 'deep_stack_cnn_lstm']
    
    estimators = []
    
    for model_name in include_models:
        if model_name == 'cnn':
            params = OptimizedHyperparameters.CNN if use_tuned_params else {}
            estimators.append(('CNN', CNNClassifier(random_state=RANDOM_SEED, verbose=0, **params)))
        
        elif model_name == 'lstm':
            params = OptimizedHyperparameters.LSTM if use_tuned_params else {}
            estimators.append(('LSTM', LSTMClassifier(random_state=RANDOM_SEED, verbose=0, **params)))
        
        elif model_name == 'bilstm':
            params = OptimizedHyperparameters.BiLSTM if use_tuned_params else {}
            estimators.append(('BiLSTM', LSTMClassifier(random_state=RANDOM_SEED, verbose=0, **params)))
        
        elif model_name == 'gru':
            params = OptimizedHyperparameters.GRU if use_tuned_params else {}
            estimators.append(('GRU', GRUClassifier(random_state=RANDOM_SEED, verbose=0, **params)))
        
        elif model_name == 'cnn_lstm':
            params = OptimizedHyperparameters.CNN_LSTM if use_tuned_params else {}
            estimators.append(('CNN-LSTM', CNNLSTMClassifier(random_state=RANDOM_SEED, verbose=0, **params)))
        
        elif model_name == 'attention_lstm':
            params = OptimizedHyperparameters.ATTENTION_LSTM if use_tuned_params else {}
            estimators.append(('Attention-LSTM', AttentionLSTMClassifier(random_state=RANDOM_SEED, verbose=0, **params)))
    
    # Create and fit ensemble
    ensemble = DeepEnsembleVoting(
        estimators=estimators,
        voting=voting,
        random_state=RANDOM_SEED,
        verbose=verbose
    )
    
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    return ensemble
