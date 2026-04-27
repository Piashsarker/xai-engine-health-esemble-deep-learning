# ============================================================
# DynamicClassifierSelector Component
# ============================================================
# Purpose: Automatically select optimal classifier combinations
# for stacking based on diversity and performance metrics
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
from .config import RANDOM_SEED


class DynamicClassifierSelector:
    """
    DynamicClassifierSelector class for automatically selecting optimal
    classifier combinations for stacked ensemble models.
    
    This class implements several research-based techniques:
    - Diversity-based selection using Q-statistic and disagreement measures
    - Performance-based selection using cross-validation accuracy
    - Greedy forward selection for iterative ensemble building
    - Combined approach using both diversity and performance
    """
    
    def __init__(self, cv: int = 5, random_state: int = RANDOM_SEED):
        """
        Initialize the DynamicClassifierSelector.
        
        Args:
            cv (int): Number of cross-validation folds (default: 5)
            random_state (int): Random seed for reproducibility
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
                                      eval_metric='mlogloss')
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
        
        print("[...] Fitting all classifiers...")
        
        for name, clf in self.classifiers.items():
            clf_clone = clone(clf)
            clf_clone.fit(X_train, y_train)
            self.predictions[name] = clf_clone.predict(X_val)
            self.accuracies[name] = accuracy_score(y_val, self.predictions[name])
            print(f"   {name}: {self.accuracies[name]:.4f}")
        
        # Display ranked performance
        print("\n[CHART] Individual Classifier Performance (Ranked):")
        print("-"*50)
        ranked = sorted(self.accuracies.items(), key=lambda x: x[1], reverse=True)
        for i, (name, acc) in enumerate(ranked, 1):
            bar = '*' * int(acc * 50)
            print(f"   {i:2d}. {name:<10} {acc:.4f} |{bar}")
        
        print(f"\n[OK] All {len(self.accuracies)} classifiers fitted and evaluated!")
        
        return self.accuracies
    
    def calculate_q_statistic(self, pred1: np.ndarray, pred2: np.ndarray, 
                               y_true: np.ndarray) -> float:
        """
        Calculate Q-statistic (Yule's Q) between two classifiers.
        
        Q ranges from -1 to 1:
        - Q = 1: Classifiers always agree
        - Q = 0: Classifiers are independent
        - Q = -1: Classifiers always disagree (maximum diversity)
        """
        correct1 = (pred1 == y_true)
        correct2 = (pred2 == y_true)
        
        n11 = np.sum(correct1 & correct2)
        n00 = np.sum(~correct1 & ~correct2)
        n10 = np.sum(correct1 & ~correct2)
        n01 = np.sum(~correct1 & correct2)
        
        numerator = n11 * n00 - n01 * n10
        denominator = n11 * n00 + n01 * n10
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_disagreement(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """Calculate disagreement measure between two classifiers."""
        return np.mean(pred1 != pred2)
    
    def calculate_diversity_matrix(self, y_true: np.ndarray) -> pd.DataFrame:
        """Calculate pairwise diversity matrix using Q-statistic."""
        print("\nQ-Statistic interpretation:")
        print("   Q = 1.0  : Classifiers always agree (low diversity)")
        print("   Q = 0.0  : Classifiers are independent (good diversity)")
        print("   Q = -1.0 : Classifiers always disagree (maximum diversity)")
        print("\nLower Q values between classifiers = Better for ensemble!")
        
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
        
        print("\n[OK] Diversity matrix calculated!")
        print("\n[TABLE] Q-Statistic Diversity Matrix:")
        print(self.diversity_matrix.round(4).to_string())
        
        return self.diversity_matrix
    
    def select_by_performance(self, top_k: int = 5) -> list:
        """Select top-k classifiers based on individual accuracy."""
        sorted_classifiers = sorted(self.accuracies.items(), 
                                    key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_classifiers[:top_k]]
    
    def select_by_diversity(self, y_true: np.ndarray, top_k: int = 5) -> list:
        """Select classifiers that maximize diversity (minimize average Q-statistic)."""
        if self.diversity_matrix is None:
            self.calculate_diversity_matrix(y_true)
        
        names = list(self.predictions.keys())
        selected = []
        
        best = max(self.accuracies.items(), key=lambda x: x[1])[0]
        selected.append(best)
        
        while len(selected) < top_k and len(selected) < len(names):
            best_candidate = None
            best_avg_q = float('inf')
            
            for name in names:
                if name in selected:
                    continue
                
                avg_q = np.mean([self.diversity_matrix.loc[name, s] for s in selected])
                
                if avg_q < best_avg_q:
                    best_avg_q = avg_q
                    best_candidate = name
            
            if best_candidate:
                selected.append(best_candidate)
        
        return selected
    
    def select_combined(self, y_true: np.ndarray, top_k: int = 5, 
                        performance_weight: float = 0.6) -> list:
        """Select classifiers using combined performance and diversity score."""
        if self.diversity_matrix is None:
            self.calculate_diversity_matrix(y_true)
        
        names = list(self.predictions.keys())
        selected = []
        
        best = max(self.accuracies.items(), key=lambda x: x[1])[0]
        selected.append(best)
        
        while len(selected) < top_k and len(selected) < len(names):
            best_candidate = None
            best_score = -float('inf')
            
            for name in names:
                if name in selected:
                    continue
                
                perf_score = self.accuracies[name]
                avg_q = np.mean([self.diversity_matrix.loc[name, s] for s in selected])
                div_score = 1 - avg_q
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
        """Greedy forward selection: iteratively add classifiers that improve ensemble."""
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
                
                test_selected = selected + [name]
                estimators = [(n, clone(self.classifiers[n])) for n in test_selected]
                stack = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
                    cv=self.cv,
                    n_jobs=-1
                )
                
                stack.fit(X_train, y_train)
                accuracy = stack.score(X_val, y_val)
                
                if accuracy > best_new_accuracy:
                    best_new_accuracy = accuracy
                    best_candidate = name
            
            if best_candidate is None or best_new_accuracy <= best_accuracy:
                print(f"   Stopping: No improvement found")
                break
            
            selected.append(best_candidate)
            best_accuracy = best_new_accuracy
            print(f"   Added {best_candidate}: Ensemble accuracy = {best_accuracy:.4f}")
        
        return selected
    
    def create_dynamic_stack(self, selected_classifiers: list) -> StackingClassifier:
        """Create a StackingClassifier from selected classifiers."""
        estimators = [(name.lower(), clone(self.classifiers[name])) 
                      for name in selected_classifiers]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=self.cv,
            n_jobs=-1
        )
    
    def plot_diversity_heatmap(self) -> None:
        """Plot heatmap of classifier diversity (Q-statistic matrix)."""
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
        """Display summary of classifier selection."""
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
        """Run all selection methods and return results."""
        if X_val is None:
            X_val = X_train
            y_val = y_train
        
        self.fit_all_classifiers(X_train, y_train, X_val, y_val)
        self.calculate_diversity_matrix(y_val)
        
        results = {}
        
        print("\n" + "-"*70)
        print("Running All Selection Methods (top_k={})".format(top_k))
        print("-"*70)
        
        # Method 1: Performance-Based Selection
        print("\n[1] PERFORMANCE-BASED SELECTION")
        print("    Strategy: Select top-{} classifiers by individual accuracy".format(top_k))
        perf_selected = self.select_by_performance(top_k)
        results['performance'] = perf_selected
        self.display_selection_summary('Performance-Based', perf_selected)
        
        # Method 2: Diversity-Based Selection
        print("\n[2] DIVERSITY-BASED SELECTION")
        print("    Strategy: Maximize ensemble diversity using Q-statistic")
        div_selected = self.select_by_diversity(y_val, top_k)
        results['diversity'] = div_selected
        self.display_selection_summary('Diversity-Based', div_selected)
        
        # Method 3: Combined Selection
        print("\n[3] COMBINED SELECTION (RECOMMENDED)")
        print("    Strategy: Balance performance (60%) + diversity (40%)")
        comb_selected = self.select_combined(y_val, top_k)
        results['combined'] = comb_selected
        self.display_selection_summary('Combined (60% Perf + 40% Div)', comb_selected)
        
        print("\n" + "="*70)
        print("[OK] All selection methods completed!")
        print("="*70)
        
        self.plot_diversity_heatmap()
        
        return results
    
    def display_results_summary(self, evaluation_results: dict) -> pd.DataFrame:
        """
        Display comprehensive results summary for dynamic classifier selection.
        
        Args:
            evaluation_results: Dictionary with model names as keys and metrics as values
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        print("\n" + "="*70)
        print("DYNAMIC CLASSIFIER SELECTION - RESULTS SUMMARY")
        print("="*70)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Classifiers': ', '.join(results.get('Classifiers', [])),
                'Accuracy': results.get('Accuracy', 0),
                'Precision': results.get('Precision', 0),
                'AUC': results.get('AUC', 0),
                'RMSE': results.get('RMSE', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display results table
        print("\n[TABLE] Dynamic Stacked Model Performance:")
        print("-"*90)
        print(f"{'Model':<24} {'Accuracy':>10} {'Precision':>10} {'AUC':>10} {'RMSE':>10}")
        print("-"*90)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['Model']:<24} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} {row['AUC']:>10.4f} {row['RMSE']:>10.4f}")
        
        print("-"*90)
        
        # Find best model
        if len(comparison_df) > 0:
            best_idx = comparison_df['Accuracy'].idxmax()
            best_model = comparison_df.loc[best_idx]
            
            print(f"\n[STAR] Best Dynamic Stack: {best_model['Model']}")
            print(f"       Accuracy: {best_model['Accuracy']:.4f}")
            print(f"       Classifiers: {best_model['Classifiers']}")
        
        # Display classifier selections
        print("\n[LIST] Classifier Selections by Method:")
        print("-"*70)
        for _, row in comparison_df.iterrows():
            method = row['Model'].replace('DS-Stack ', '')
            print(f"   {method:<15}: {row['Classifiers']}")
        
        return comparison_df
