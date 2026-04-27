# ============================================================
# Deep Stacked Ensemble Module
# ============================================================
# Purpose: Stacked ensemble with deep learning meta-learners
# ============================================================

import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .config import RANDOM_SEED


class DeepStackedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacked Ensemble with Deep Learning meta-learner.
    
    Uses traditional ML or deep learning classifiers as base models,
    and a deep learning model (CNN, LSTM, or CNN-LSTM) as the meta-learner
    to combine base predictions.
    
    Architecture:
    - Level 0: Base classifiers generate out-of-fold predictions
    - Level 1: Meta-learner (CNN/LSTM/CNN-LSTM) combines predictions
    
    Parameters:
    -----------
    base_classifiers : list of tuples
        List of (name, classifier) tuples for base models
    meta_learner_type : str
        Type of meta-learner: 'cnn', 'lstm', 'cnn_lstm', 'bilstm', 
        'attention_lstm', 'gru', or 'lr' (logistic regression)
    cv : int
        Number of cross-validation folds for generating meta-features
    use_probabilities : bool
        Whether to use probability predictions (True) or class predictions (False)
    passthrough : bool
        Whether to include original features along with meta-features
    random_state : int
        Random seed for reproducibility
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    """
    
    def __init__(self, base_classifiers, meta_learner_type='cnn_lstm',
                 cv=5, use_probabilities=True, passthrough=False,
                 random_state=RANDOM_SEED, verbose=1):
        self.base_classifiers = base_classifiers
        self.meta_learner_type = meta_learner_type
        self.cv = cv
        self.use_probabilities = use_probabilities
        self.passthrough = passthrough
        self.random_state = random_state
        self.verbose = verbose
        self.trained_base = []
        self.meta_learner = None
        self.n_classes = None
        self.classes_ = None
        self.training_time = None
    
    def _create_meta_learner(self, n_meta_features):
        """Create the meta-learner based on specified type."""
        # Import here to avoid circular imports
        from .deep_learning_classifiers import (
            CNNClassifier, LSTMClassifier, CNNLSTMClassifier,
            GRUClassifier, AttentionLSTMClassifier
        )
        
        if self.meta_learner_type == 'cnn':
            return CNNClassifier(
                n_features=n_meta_features,
                n_classes=self.n_classes,
                filters=[32, 64, 128],
                dropout_rate=0.3,
                epochs=100,
                random_state=self.random_state,
                verbose=0
            )
        elif self.meta_learner_type == 'lstm':
            return LSTMClassifier(
                n_features=n_meta_features,
                n_classes=self.n_classes,
                lstm_units=[32, 16],
                dropout_rate=0.3,
                epochs=100,
                random_state=self.random_state,
                verbose=0
            )
        elif self.meta_learner_type == 'bilstm':
            return LSTMClassifier(
                n_features=n_meta_features,
                n_classes=self.n_classes,
                lstm_units=[32, 16],
                dropout_rate=0.3,
                bidirectional=True,
                epochs=100,
                random_state=self.random_state,
                verbose=0
            )
        elif self.meta_learner_type == 'cnn_lstm':
            return CNNLSTMClassifier(
                n_features=n_meta_features,
                n_classes=self.n_classes,
                cnn_filters=[32, 64],
                lstm_units=[32],
                dropout_rate=0.3,
                epochs=100,
                random_state=self.random_state,
                verbose=0
            )
        elif self.meta_learner_type == 'gru':
            return GRUClassifier(
                n_features=n_meta_features,
                n_classes=self.n_classes,
                gru_units=[32, 16],
                dropout_rate=0.3,
                epochs=100,
                random_state=self.random_state,
                verbose=0
            )
        elif self.meta_learner_type == 'attention_lstm':
            return AttentionLSTMClassifier(
                n_features=n_meta_features,
                n_classes=self.n_classes,
                lstm_units=[32, 32],
                attention_units=32,
                dropout_rate=0.3,
                epochs=100,
                random_state=self.random_state,
                verbose=0
            )
        else:  # Default to logistic regression
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )

    def _get_oof_predictions(self, clf, X, y, name):
        """Generate out-of-fold predictions for a classifier."""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, 
                              random_state=self.random_state)
        
        if self.use_probabilities and hasattr(clf, 'predict_proba'):
            # Use probability predictions
            oof_preds = np.zeros((X.shape[0], self.n_classes))
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                clf_clone = clone(clf)
                clf_clone.fit(X_train_fold, y_train_fold)
                oof_preds[val_idx] = clf_clone.predict_proba(X_val_fold)
        else:
            # Use class predictions (one-hot encoded)
            oof_preds = np.zeros((X.shape[0], self.n_classes))
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                clf_clone = clone(clf)
                clf_clone.fit(X_train_fold, y_train_fold)
                preds = clf_clone.predict(X_val_fold)
                oof_preds[val_idx] = np.eye(self.n_classes)[preds]
        
        return oof_preds
    
    def fit(self, X, y):
        """
        Fit the deep stacked ensemble.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Fitted ensemble
        """
        start_time = time.time()
        
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("DEEP STACKED ENSEMBLE TRAINING")
            print("="*60)
            print(f"\nBase Classifiers: {len(self.base_classifiers)}")
            print(f"Meta-Learner: {self.meta_learner_type.upper()}")
            print(f"CV Folds: {self.cv}")
            print(f"Use Probabilities: {self.use_probabilities}")
            print(f"Passthrough: {self.passthrough}")
        
        # Generate meta-features using cross-validation
        meta_features = []
        self.trained_base = []
        
        for name, clf in self.base_classifiers:
            if self.verbose >= 1:
                print(f"\n[...] Training base classifier: {name}")
            
            # Get out-of-fold predictions
            oof_preds = self._get_oof_predictions(clf, X, y, name)
            meta_features.append(oof_preds)
            
            # Fit on full training data
            clf_clone = clone(clf)
            clf_clone.fit(X, y)
            self.trained_base.append((name, clf_clone))
            
            # Calculate and display accuracy
            train_acc = accuracy_score(y, clf_clone.predict(X))
            if self.verbose >= 1:
                print(f"   [OK] {name} trained - Train Accuracy: {train_acc:.4f}")
        
        # Stack meta-features
        X_meta = np.hstack(meta_features)
        
        # Optionally include original features
        if self.passthrough:
            X_meta = np.hstack([X, X_meta])
        
        if self.verbose >= 1:
            print(f"\n[INFO] Meta-features shape: {X_meta.shape}")
            print(f"\n[...] Training {self.meta_learner_type.upper()} meta-learner...")
        
        # Train meta-learner
        self.meta_learner = self._create_meta_learner(X_meta.shape[1])
        self.meta_learner.fit(X_meta, y)
        
        self.training_time = time.time() - start_time
        
        if self.verbose >= 1:
            meta_train_acc = accuracy_score(y, self.meta_learner.predict(X_meta))
            print(f"   [OK] Meta-learner trained - Train Accuracy: {meta_train_acc:.4f}")
            print(f"\n[OK] Deep Stacked Ensemble trained in {self.training_time:.2f} seconds")
            print("="*60)
        
        return self
    
    def _generate_meta_features(self, X):
        """Generate meta-features for prediction."""
        meta_features = []
        
        for name, clf in self.trained_base:
            if self.use_probabilities and hasattr(clf, 'predict_proba'):
                meta_features.append(clf.predict_proba(X))
            else:
                preds = clf.predict(X)
                meta_features.append(np.eye(self.n_classes)[preds])
        
        X_meta = np.hstack(meta_features)
        
        if self.passthrough:
            X_meta = np.hstack([X, X_meta])
        
        return X_meta
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.meta_learner is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        X_meta = self._generate_meta_features(X)
        return self.meta_learner.predict(X_meta)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        if self.meta_learner is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        X_meta = self._generate_meta_features(X)
        return self.meta_learner.predict_proba(X_meta)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return accuracy_score(y, self.predict(X))
    
    def get_base_predictions(self, X):
        """Get predictions from all base classifiers."""
        predictions = {}
        for name, clf in self.trained_base:
            predictions[name] = clf.predict(X)
        return predictions
    
    def get_base_accuracies(self, X, y):
        """Get accuracies of all base classifiers."""
        accuracies = {}
        for name, clf in self.trained_base:
            accuracies[name] = accuracy_score(y, clf.predict(X))
        return accuracies

    def display_architecture(self):
        """Display the architecture of the deep stacked ensemble."""
        if self.meta_learner is None:
            print("[!] Ensemble not fitted. Call fit() first.")
            return
        
        base_names = [name.upper() for name, _ in self.trained_base]
        
        print("\n" + "="*70)
        print("DEEP STACKED ENSEMBLE ARCHITECTURE")
        print("="*70)
        
        print("\n" + " "*20 + "+" + "-"*28 + "+")
        print(" "*20 + "|" + "    Input Features (9)     " + "|")
        print(" "*20 + "+" + "-"*28 + "+")
        print(" "*33 + "|")
        print(" "*33 + "v")
        
        print("\n" + " "*5 + "+" + "-"*60 + "+")
        print(" "*5 + "|" + " "*15 + "Level 0: Base Classifiers" + " "*20 + "|")
        print(" "*5 + "|" + "-"*60 + "|")
        
        # Display base classifiers
        classifier_line = "|  "
        for name in base_names:
            classifier_line += f"[{name:^8}]  "
        classifier_line = classifier_line.ljust(65) + "|"
        print(" "*5 + classifier_line)
        
        print(" "*5 + "+" + "-"*60 + "+")
        print(" "*33 + "|")
        print(" "*25 + "(Probability Predictions)")
        print(" "*33 + "|")
        print(" "*33 + "v")
        
        print("\n" + " "*10 + "+" + "-"*48 + "+")
        print(" "*10 + "|" + "    Level 1: Deep Learning Meta-Learner        " + "|")
        print(" "*10 + "|" + f"    {self.meta_learner_type.upper():^42} " + "|")
        print(" "*10 + "+" + "-"*48 + "+")
        print(" "*33 + "|")
        print(" "*33 + "v")
        
        print("\n" + " "*15 + "+" + "-"*38 + "+")
        print(" "*15 + "|" + "        Final Prediction             " + "|")
        print(" "*15 + "|" + " Good | Minor | Moderate | Critical " + "|")
        print(" "*15 + "+" + "-"*38 + "+")
        
        print("\n" + "-"*70)
        print("Model Details:")
        print(f"  - Base Classifiers: {', '.join(base_names)}")
        print(f"  - Number of Base Classifiers: {len(base_names)}")
        print(f"  - Meta-Learner: {self.meta_learner_type.upper()}")
        print(f"  - Cross-Validation Folds: {self.cv}")
        print(f"  - Use Probabilities: {self.use_probabilities}")
        print(f"  - Passthrough: {self.passthrough}")
        if self.training_time:
            print(f"  - Training Time: {self.training_time:.2f} seconds")
        print("-"*70)


class DynamicDeepStackedEnsemble(DeepStackedEnsemble):
    """
    Dynamic Deep Stacked Ensemble with automatic classifier selection.
    
    Extends DeepStackedEnsemble with dynamic selection of base classifiers
    based on performance and diversity metrics.
    
    Parameters:
    -----------
    selection_method : str
        Method for selecting base classifiers:
        'performance' - Select top-k by accuracy
        'diversity' - Select for maximum diversity
        'combined' - Balance performance and diversity
        'greedy' - Greedy forward selection
    top_k : int
        Number of base classifiers to select
    performance_weight : float
        Weight for performance in combined selection (0-1)
    **kwargs : dict
        Additional arguments passed to DeepStackedEnsemble
    """
    
    def __init__(self, selection_method='combined', top_k=5,
                 performance_weight=0.6, **kwargs):
        super().__init__(base_classifiers=[], **kwargs)
        self.selection_method = selection_method
        self.top_k = top_k
        self.performance_weight = performance_weight
        self.selector = None
        self.selected_classifiers = None
    
    def fit(self, X, y):
        """
        Fit the dynamic deep stacked ensemble.
        
        Automatically selects optimal base classifiers before training.
        """
        from .dynamic_classifier_selector import DynamicClassifierSelector
        
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("DYNAMIC DEEP STACKED ENSEMBLE")
            print("="*60)
            print(f"\nSelection Method: {self.selection_method}")
            print(f"Top-K Classifiers: {self.top_k}")
        
        # Initialize selector and fit all classifiers
        self.selector = DynamicClassifierSelector(
            cv=self.cv,
            random_state=self.random_state
        )
        
        # Split data for selection
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Fit all classifiers and get predictions
        self.selector.fit_all_classifiers(X_train, y_train, X_val, y_val)
        self.selector.calculate_diversity_matrix(y_val)
        
        # Select classifiers based on method
        if self.selection_method == 'performance':
            self.selected_classifiers = self.selector.select_by_performance(self.top_k)
        elif self.selection_method == 'diversity':
            self.selected_classifiers = self.selector.select_by_diversity(y_val, self.top_k)
        elif self.selection_method == 'combined':
            self.selected_classifiers = self.selector.select_combined(
                y_val, self.top_k, self.performance_weight
            )
        else:
            self.selected_classifiers = self.selector.select_by_performance(self.top_k)
        
        if self.verbose >= 1:
            print(f"\n[OK] Selected Classifiers: {self.selected_classifiers}")
        
        # Create base classifiers list
        self.base_classifiers = [
            (name, clone(self.selector.classifiers[name]))
            for name in self.selected_classifiers
        ]
        
        # Call parent fit method
        return super().fit(X, y)
    
    def display_selection_summary(self):
        """Display summary of classifier selection."""
        if self.selected_classifiers is None:
            print("[!] Ensemble not fitted. Call fit() first.")
            return
        
        print("\n" + "="*60)
        print(f"DYNAMIC SELECTION SUMMARY: {self.selection_method.upper()}")
        print("="*60)
        print(f"\nSelected Classifiers ({len(self.selected_classifiers)}):")
        
        for i, name in enumerate(self.selected_classifiers, 1):
            acc = self.selector.accuracies.get(name, 'N/A')
            if isinstance(acc, float):
                print(f"   {i}. {name} (Accuracy: {acc:.4f})")
            else:
                print(f"   {i}. {name}")
        
        print("-"*60)
