# ============================================================
# DynamicClassifierSelector Component
# ============================================================
# Purpose: Automatically select optimal classifier combinations
# for stacking based on diversity and performance metrics
# 
# Research-based techniques implemented:
# 1. Diversity-based selection using Q-statistic
# 2. Performance-based selection using accuracy
# 3. Greedy forward selection
# 4. Combined approach (diversity + performance)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, StackingClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class DynamicClassifierSelector:
    """
    DynamicClassifierSelector class for automatically selecting optimal
    classifier combinations for stacked ensemble models.
    
    This class implements several research-based techniques:
    - Diversity-based selection using Q-statistic and disagreement measures
    - Performance-based selection using cross-validation accuracy
    - Greedy forward selection for iterative ensemble building
    - Combined approach using both diversity and performance
    
    Attributes:
        classifiers (dict): Dictionary of available classifiers
        cv (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, cv: int = 5, random_state: int = 42):
        """
        Initialize the DynamicClassifierSelector.
        
        Args:
            cv (int): Number of cross-validation folds (default: 5)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        self.cv = cv
        self.random_state = random_state
        
        # Initialize all available classifiers
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
            'XGBoost': XGBClassifier(n_estimators=100, random_state=random_state,
                                      use_label_encoder=False, eval_metric='mlogloss')
        }
        
        self.predictions = {}
        self.accuracies = {}
        self.diversity_matrix = None
    
    def fit_all_classifiers(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Fit all classifiers and store their predictions.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, uses X_train if not provided)
            y_val: Validation labels (optional, uses y_train if not provided)
            
        Returns:
            dict: Dictionary of classifier accuracies
        """
        if X_val is None:
            X_val = X_train
            y_val = y_train
        
        print("[...] Fitting all classifiers for dynamic selection...")
        
        for name, clf in self.classifiers.items():
            # Clone classifier to avoid modifying original
            clf_clone = clone(clf)
            clf_clone.fit(X_train, y_train)
            
            # Store predictions
            self.predictions[name] = clf_clone.predict(X_val)
            
            # Calculate accuracy
            self.accuracies[name] = accuracy_score(y_val, self.predictions[name])
            print(f"   {name}: {self.accuracies[name]:.4f}")
        
        return self.accuracies
    
    def calculate_q_statistic(self, pred1: np.ndarray, pred2: np.ndarray, 
                               y_true: np.ndarray) -> float:
        """
        Calculate Q-statistic (Yule's Q) between two classifiers.
        
        Q ranges from -1 to 1:
        - Q = 1: Classifiers always agree
        - Q = 0: Classifiers are independent
        - Q = -1: Classifiers always disagree (maximum diversity)
        
        Lower Q values indicate higher diversity (better for ensembles).
        """
        # Calculate contingency table
        correct1 = (pred1 == y_true)
        correct2 = (pred2 == y_true)
        
        n11 = np.sum(correct1 & correct2)      # Both correct
        n00 = np.sum(~correct1 & ~correct2)    # Both wrong
        n10 = np.sum(correct1 & ~correct2)     # 1 correct, 2 wrong
        n01 = np.sum(~correct1 & correct2)     # 1 wrong, 2 correct
        
        # Calculate Q-statistic
        numerator = n11 * n00 - n01 * n10
        denominator = n11 * n00 + n01 * n10
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_disagreement(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """
        Calculate disagreement measure between two classifiers.
        
        Disagreement = proportion of samples where classifiers disagree.
        Higher disagreement indicates higher diversity.
        """
        return np.mean(pred1 != pred2)
    
    def calculate_diversity_matrix(self, y_true: np.ndarray) -> pd.DataFrame:
        """
        Calculate pairwise diversity matrix using Q-statistic.
        
        Returns:
            pd.DataFrame: Matrix of Q-statistics between all classifier pairs
        """
        names = list(self.predictions.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self.calculate_q_statistic(
                        self.predictions[name1], 
                        self.predictions[name2], 
                        y_true
                    )
        
        self.diversity_matrix = pd.DataFrame(matrix, index=names, columns=names)
        return self.diversity_matrix
    
    def select_by_performance(self, top_k: int = 5) -> list:
        """
        Select top-k classifiers based on individual accuracy.
        
        Args:
            top_k (int): Number of top classifiers to select
            
        Returns:
            list: Names of top-k classifiers
        """
        sorted_classifiers = sorted(self.accuracies.items(), 
                                    key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_classifiers[:top_k]]
    
    def select_by_diversity(self, y_true: np.ndarray, top_k: int = 5) -> list:
        """
        Select classifiers that maximize diversity (minimize average Q-statistic).
        
        Uses greedy selection: starts with best classifier, then adds
        classifiers that have lowest average Q with already selected ones.
        
        Args:
            y_true: True labels for diversity calculation
            top_k (int): Number of classifiers to select
            
        Returns:
            list: Names of selected diverse classifiers
        """
        if self.diversity_matrix is None:
            self.calculate_diversity_matrix(y_true)
        
        names = list(self.predictions.keys())
        selected = []
        
        # Start with best performing classifier
        best = max(self.accuracies.items(), key=lambda x: x[1])[0]
        selected.append(best)
        
        # Greedily add classifiers with lowest average Q to selected set
        while len(selected) < top_k and len(selected) < len(names):
            best_candidate = None
            best_avg_q = float('inf')
            
            for name in names:
                if name in selected:
                    continue
                
                # Calculate average Q with already selected classifiers
                avg_q = np.mean([self.diversity_matrix.loc[name, s] for s in selected])
                
                if avg_q < best_avg_q:
                    best_avg_q = avg_q
                    best_candidate = name
            
            if best_candidate:
                selected.append(best_candidate)
        
        return selected
    
    def select_combined(self, y_true: np.ndarray, top_k: int = 5, 
                        performance_weight: float = 0.6) -> list:
        """
        Select classifiers using combined performance and diversity score.
        
        Combined Score = performance_weight * accuracy + (1 - performance_weight) * (1 - avg_Q)
        
        Args:
            y_true: True labels for diversity calculation
            top_k (int): Number of classifiers to select
            performance_weight (float): Weight for performance vs diversity (0-1)
            
        Returns:
            list: Names of selected classifiers
        """
        if self.diversity_matrix is None:
            self.calculate_diversity_matrix(y_true)
        
        names = list(self.predictions.keys())
        selected = []
        
        # Start with best performing classifier
        best = max(self.accuracies.items(), key=lambda x: x[1])[0]
        selected.append(best)
        
        while len(selected) < top_k and len(selected) < len(names):
            best_candidate = None
            best_score = -float('inf')
            
            for name in names:
                if name in selected:
                    continue
                
                # Performance component (normalized)
                perf_score = self.accuracies[name]
                
                # Diversity component (1 - avg_Q, so higher is more diverse)
                avg_q = np.mean([self.diversity_matrix.loc[name, s] for s in selected])
                div_score = 1 - avg_q
                
                # Combined score
                combined = performance_weight * perf_score + (1 - performance_weight) * div_score
                
                if combined > best_score:
                    best_score = combined
                    best_candidate = name
            
            if best_candidate:
                selected.append(best_candidate)
        
        return selected
    
    def greedy_forward_selection(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  max_classifiers: int = 5) -> list:
        """
        Greedy forward selection: iteratively add classifiers that improve ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            max_classifiers (int): Maximum number of classifiers to select
            
        Returns:
            list: Names of selected classifiers
        """
        names = list(self.classifiers.keys())
        selected = []
        best_accuracy = 0
        
        print("\n[...] Running greedy forward selection...")
        
        while len(selected) < max_classifiers:
            best_candidate = None
            best_new_accuracy = best_accuracy
            
            for name in names:
                if name in selected:
                    continue
                
                # Try adding this classifier
                test_selected = selected + [name]
                
                # Build stacking classifier with current selection
                estimators = [(n, clone(self.classifiers[n])) for n in test_selected]
                stack = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
                    cv=self.cv,
                    n_jobs=-1
                )
                
                # Evaluate
                stack.fit(X_train, y_train)
                accuracy = stack.score(X_val, y_val)
                
                if accuracy > best_new_accuracy:
                    best_new_accuracy = accuracy
                    best_candidate = name
            
            # If no improvement, stop
            if best_candidate is None or best_new_accuracy <= best_accuracy:
                print(f"   Stopping: No improvement found")
                break
            
            selected.append(best_candidate)
            best_accuracy = best_new_accuracy
            print(f"   Added {best_candidate}: Ensemble accuracy = {best_accuracy:.4f}")
        
        return selected
    
    def create_dynamic_stack(self, selected_classifiers: list) -> StackingClassifier:
        """
        Create a StackingClassifier from selected classifiers.
        
        Args:
            selected_classifiers (list): Names of classifiers to include
            
        Returns:
            StackingClassifier: Configured stacking classifier
        """
        estimators = [(name.lower(), clone(self.classifiers[name])) 
                      for name in selected_classifiers]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=self.cv,
            n_jobs=-1
        )
    
    def plot_diversity_heatmap(self) -> None:
        """
        Plot heatmap of classifier diversity (Q-statistic matrix).
        """
        if self.diversity_matrix is None:
            print("[!] Calculate diversity matrix first using calculate_diversity_matrix()")
            return
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.diversity_matrix, annot=True, fmt='.3f', 
                    cmap='RdYlGn_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5)
        plt.title('Classifier Diversity Matrix (Q-Statistic)\n(Lower values = Higher diversity)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def display_selection_summary(self, method: str, selected: list) -> None:
        """
        Display summary of classifier selection.
        """
        print(f"\n" + "="*60)
        print(f"DYNAMIC CLASSIFIER SELECTION: {method.upper()}")
        print("="*60)
        print(f"\nSelected Classifiers ({len(selected)}):")
        for i, name in enumerate(selected, 1):
            acc = self.accuracies.get(name, 'N/A')
            if isinstance(acc, float):
                print(f"   {i}. {name} (Accuracy: {acc:.4f})")
            else:
                print(f"   {i}. {name}")
        print("-"*60)
    
    def run_all_selection_methods(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                                   top_k: int = 5) -> dict:
        """
        Run all selection methods and return results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            top_k: Number of classifiers to select
            
        Returns:
            dict: Dictionary with selection results from each method
        """
        if X_val is None:
            X_val = X_train
            y_val = y_train
        
        # Fit all classifiers first
        self.fit_all_classifiers(X_train, y_train, X_val, y_val)
        
        # Calculate diversity matrix
        self.calculate_diversity_matrix(y_val)
        
        results = {}
        
        # Method 1: Performance-based
        perf_selected = self.select_by_performance(top_k)
        results['performance'] = perf_selected
        self.display_selection_summary('Performance-Based', perf_selected)
        
        # Method 2: Diversity-based
        div_selected = self.select_by_diversity(y_val, top_k)
        results['diversity'] = div_selected
        self.display_selection_summary('Diversity-Based', div_selected)
        
        # Method 3: Combined
        comb_selected = self.select_combined(y_val, top_k)
        results['combined'] = comb_selected
        self.display_selection_summary('Combined (60% Perf, 40% Div)', comb_selected)
        
        # Plot diversity heatmap
        self.plot_diversity_heatmap()
        
        return results


# Example usage (can be run in notebook):
"""
# Initialize selector
selector = DynamicClassifierSelector(cv=5, random_state=42)

# Run all selection methods
results = selector.run_all_selection_methods(X_train, y_train, X_test, y_test, top_k=5)

# Create and train dynamic stack from combined selection
dynamic_stack = selector.create_dynamic_stack(results['combined'])
dynamic_stack.fit(X_train, y_train)
print(f"Dynamic Stack Accuracy: {dynamic_stack.score(X_test, y_test):.4f}")

# Or run greedy forward selection (slower but more accurate)
greedy_selected = selector.greedy_forward_selection(X_train, y_train, X_test, y_test, max_classifiers=5)
"""
