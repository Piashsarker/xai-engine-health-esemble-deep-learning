# ============================================================
# BaseClassifierModule Component
# ============================================================
# Purpose: Train and evaluate individual base classifiers
# ============================================================

import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from .config import RANDOM_SEED


class BaseClassifierModule:
    """
    BaseClassifierModule class for training and evaluating individual base classifiers.
    
    This class provides methods to:
    - Initialize 10 different classifiers (LR, KNN, LDA, GNB, SVM, DT, RF, AdaBoost, GB, XGBoost)
    - Train individual classifiers on training data
    - Train all classifiers at once
    - Generate predictions and probability predictions
    - Retrieve trained classifier instances
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialize the BaseClassifierModule with 10 base classifiers.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        
        self.classifiers = {
            'LR': LogisticRegression(max_iter=1000, random_state=random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'LDA': LinearDiscriminantAnalysis(),
            'GNB': GaussianNB(),
            'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
            'DT': DecisionTreeClassifier(random_state=random_state),
            'RF': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=random_state),
            'GB': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'XGBoost': XGBClassifier(
                n_estimators=100, 
                random_state=random_state,
                eval_metric='mlogloss'
            )
        }
        
        self.trained_models = {}
        self.training_results = {}
        
        print("="*60)
        print("BASE CLASSIFIER MODULE INITIALIZED")
        print("="*60)
        print(f"\nRandom State: {random_state}")
        print(f"\nAvailable Classifiers: {len(self.classifiers)}")
        print("-" * 50)
        for i, name in enumerate(self.classifiers.keys(), 1):
            print(f"  {i:2d}. {name}")
        print("-" * 50)
    
    def train_classifier(self, name: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray = None, y_test: np.ndarray = None) -> dict:
        """
        Train a specific classifier on the training data.
        
        Args:
            name (str): Name of the classifier to train
            X_train (np.ndarray): Training feature array
            y_train (np.ndarray): Training target array
            X_test (np.ndarray, optional): Testing feature array for evaluation
            y_test (np.ndarray, optional): Testing target array for evaluation
            
        Returns:
            dict: Dictionary containing training and testing accuracy
        """
        if name not in self.classifiers:
            raise ValueError(f"Unknown classifier: {name}. Available: {list(self.classifiers.keys())}")
        
        print(f"\n[...] Training {name} classifier...")
        
        classifier = self.classifiers[name]
        
        start_time = time.time()
        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.trained_models[name] = classifier
        
        train_predictions = classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        test_accuracy = None
        if X_test is not None and y_test is not None:
            test_predictions = classifier.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_predictions)
        
        results = {
            'name': name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }
        self.training_results[name] = results
        
        print(f"[OK] {name} trained successfully!")
        print(f"   Training Time: {training_time:.2f} seconds")
        print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        if test_accuracy is not None:
            print(f"   Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return results
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray = None, y_test: np.ndarray = None) -> pd.DataFrame:
        """
        Train all base classifiers on the training data.
        
        Args:
            X_train (np.ndarray): Training feature array
            y_train (np.ndarray): Training target array
            X_test (np.ndarray, optional): Testing feature array for evaluation
            y_test (np.ndarray, optional): Testing target array for evaluation
            
        Returns:
            pd.DataFrame: DataFrame containing training results for all classifiers
        """
        print("\n" + "="*60)
        print("TRAINING ALL BASE CLASSIFIERS")
        print("="*60)
        print(f"\nTraining samples: {X_train.shape[0]:,}")
        print(f"Features: {X_train.shape[1]}")
        if X_test is not None:
            print(f"Testing samples: {X_test.shape[0]:,}")
        
        for name in self.classifiers.keys():
            self.train_classifier(name, X_train, y_train, X_test, y_test)
        
        results_df = pd.DataFrame([
            {
                'Classifier': name,
                'Train Accuracy': results['train_accuracy'],
                'Test Accuracy': results['test_accuracy'],
                'Training Time (s)': results['training_time']
            }
            for name, results in self.training_results.items()
        ])
        
        results_df = results_df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
        
        print("\n" + "="*60)
        print("ALL BASE CLASSIFIERS TRAINED SUCCESSFULLY")
        print("="*60)
        
        return results_df
    
    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        """Generate predictions using a trained classifier."""
        if name not in self.trained_models:
            raise ValueError(f"Classifier '{name}' has not been trained. Call train_classifier() first.")
        
        return self.trained_models[name].predict(X)
    
    def predict_proba(self, name: str, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions using a trained classifier."""
        if name not in self.trained_models:
            raise ValueError(f"Classifier '{name}' has not been trained. Call train_classifier() first.")
        
        classifier = self.trained_models[name]
        
        if not hasattr(classifier, 'predict_proba'):
            raise ValueError(f"Classifier '{name}' does not support probability predictions.")
        
        return classifier.predict_proba(X)
    
    def get_classifier(self, name: str):
        """Retrieve a trained classifier instance."""
        if name not in self.trained_models:
            raise ValueError(f"Classifier '{name}' has not been trained. Call train_classifier() first.")
        
        return self.trained_models[name]
    
    def get_all_classifiers(self) -> dict:
        """Retrieve all trained classifier instances."""
        return self.trained_models.copy()
