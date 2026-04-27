# ============================================================
# StackedEnsemble Component
# ============================================================
# Purpose: Construct stacked ensemble models with different configurations
# ============================================================

import time
import numpy as np
from sklearn.ensemble import (
    StackingClassifier, RandomForestClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .config import RANDOM_SEED


class StackedEnsemble:
    """
    StackedEnsemble class for constructing stacked ensemble models.
    
    Stacked Model Configurations:
    - Stacked Model 1: KNN + SVM + RF + AdaBoost + XGBoost (all 5 base classifiers) 
    - Stacked Model 2: RF + XGBoost + AdaBoost (3 boosting/ensemble methods)
    - Stacked Model 3: KNN + SVM + RF (3 diverse classifiers)
    """
    
    def __init__(self, cv: int = 5, random_state: int = RANDOM_SEED):
        """
        Initialize the StackedEnsemble instance.
        
        Args:
            cv (int): Number of cross-validation folds (default: 5)
            random_state (int): Random seed for reproducibility
        """
        self.cv = cv
        self.random_state = random_state
        self.stacked_models = {}
        self.training_times = {}
    
    def create_stacked_model_1(self) -> StackingClassifier:
        """Create Stacked Model 1: KNN + SVM + RF + AdaBoost + XGBoost."""
        estimators = [
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=self.random_state)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=self.random_state, 
                                  eval_metric='mlogloss'))
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=self.cv,
            n_jobs=-1
        )
    
    def create_stacked_model_2(self) -> StackingClassifier:
        """Create Stacked Model 2: RF + XGBoost + AdaBoost."""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=self.random_state,
                                  eval_metric='mlogloss')),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=self.random_state))
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=self.cv,
            n_jobs=-1
        )
    
    def create_stacked_model_3(self) -> StackingClassifier:
        """Create Stacked Model 3: KNN + SVM + RF."""
        estimators = [
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state))
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=self.cv,
            n_jobs=-1
        )
    
    def train_stacked_model(self, name: str, model: StackingClassifier, 
                            X_train: np.ndarray, y_train: np.ndarray) -> StackingClassifier:
        """
        Train a stacked ensemble model using cross-validation.
        
        Args:
            name (str): Name identifier for the stacked model
            model (StackingClassifier): Configured stacking classifier to train
            X_train (np.ndarray): Training feature array
            y_train (np.ndarray): Training target array
            
        Returns:
            StackingClassifier: Trained stacking classifier
        """
        print(f"\n[...] Training {name}...")
        print(f"   Base estimators: {[est[0] for est in model.estimators]}")
        print(f"   Meta-learner: LogisticRegression")
        print(f"   Cross-validation folds: {self.cv}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.stacked_models[name] = model
        self.training_times[name] = training_time
        
        train_accuracy = model.score(X_train, y_train)
        
        print(f"\n[OK] {name} trained successfully!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        
        return model
    
    def display_architecture(self, name: str) -> None:
        """Display the architecture of a stacked model as ASCII art."""
        if name not in self.stacked_models:
            print(f"[!] Model '{name}' not found. Train the model first.")
            return
        
        model = self.stacked_models[name]
        estimator_names = [est[0].upper() for est in model.estimators]
        
        print("\n" + "="*70)
        print(f"ARCHITECTURE: {name}")
        print("="*70)
        
        print("\n" + " "*20 + "+" + "-"*28 + "+")
        print(" "*20 + "|" + "    Input Features (9)     " + "|")
        print(" "*20 + "+" + "-"*28 + "+")
        print(" "*33 + "|")
        print(" "*33 + "v")
        
        print("\n" + " "*5 + "+" + "-"*60 + "+")
        print(" "*5 + "|" + " "*15 + "Level 0: Base Classifiers" + " "*20 + "|")
        print(" "*5 + "|" + "-"*60 + "|")
        
        classifier_line = "|  "
        for est_name in estimator_names:
            classifier_line += f"[{est_name:^8}]  "
        classifier_line = classifier_line.ljust(65) + "|"
        print(" "*5 + classifier_line)
        
        print(" "*5 + "+" + "-"*60 + "+")
        print(" "*33 + "|")
        print(" "*33 + "v")
        
        print("\n" + " "*15 + "+" + "-"*38 + "+")
        print(" "*15 + "|" + "    Level 1: Meta-Learner            " + "|")
        print(" "*15 + "|" + "    Logistic Regression              " + "|")
        print(" "*15 + "+" + "-"*38 + "+")
        print(" "*33 + "|")
        print(" "*33 + "v")
        
        print("\n" + " "*15 + "+" + "-"*38 + "+")
        print(" "*15 + "|" + "        Final Prediction             " + "|")
        print(" "*15 + "|" + " Good | Minor | Moderate | Critical " + "|")
        print(" "*15 + "+" + "-"*38 + "+")
        
        print("\n" + "-"*70)
        print(f"Model Details:")
        print(f"  - Base Classifiers: {', '.join(estimator_names)}")
        print(f"  - Number of Base Classifiers: {len(estimator_names)}")
        print(f"  - Meta-Learner: Logistic Regression (max_iter=1000)")
        print(f"  - Cross-Validation Folds: {self.cv}")
        if name in self.training_times:
            print(f"  - Training Time: {self.training_times[name]:.2f} seconds")
        print("-"*70)
    
    def get_model(self, name: str) -> StackingClassifier:
        """Retrieve a trained stacked model by name."""
        if name not in self.stacked_models:
            raise KeyError(f"Model '{name}' not found. Available models: {list(self.stacked_models.keys())}")
        return self.stacked_models[name]
    
    def get_all_models(self) -> dict:
        """Retrieve all trained stacked models."""
        return self.stacked_models
