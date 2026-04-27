# ============================================================
# ModelEvaluator Component
# ============================================================
# Purpose: Calculate comprehensive performance metrics
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
)
from .config import RANDOM_SEED, TARGET_METRICS


class ModelEvaluator:
    """
    ModelEvaluator class for calculating comprehensive performance metrics.
    
    This class provides methods to:
    - Calculate RMSE, RMSD, MAE for error metrics
    - Calculate Accuracy, Precision, AUC for classification metrics
    - Generate confusion matrices for detailed analysis
    - Perform k-fold cross-validation
    - Compare multiple models and identify the best performer
    - Verify if models meet target performance metrics
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator instance."""
        self.metrics_results = {}
        self.cv_results = {}
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_rmsd(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Deviation."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Accuracy score."""
        return accuracy_score(y_true, y_pred)
    
    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           average: str = 'weighted') -> float:
        """Calculate weighted Precision score."""
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    
    def calculate_auc(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      n_classes: int = 4) -> float:
        """Calculate AUC using one-vs-rest strategy for multi-class classification."""
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        return roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='weighted')
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Generate confusion matrix for classification results."""
        return confusion_matrix(y_true, y_pred)
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5, model_name: str = 'Model',
                       scoring: list = None) -> dict:
        """
        Perform k-fold cross-validation on a model.
        
        Args:
            model: Scikit-learn compatible model
            X (np.ndarray): Feature array
            y (np.ndarray): Target array
            cv (int): Number of cross-validation folds
            model_name (str): Name of the model for reporting
            scoring (list, optional): List of scoring metrics
            
        Returns:
            dict: Dictionary containing CV scores for each metric
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'f1_weighted']
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        self.cv_results[model_name] = cv_results
        
        return cv_results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str = 'Model') -> dict:
        """
        Calculate all metrics for a single model.
        
        Args:
            model: Trained scikit-learn compatible model
            X_test (np.ndarray): Test feature array
            y_test (np.ndarray): Test target array
            model_name (str): Name of the model for reporting
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = {
            'Model': model_name,
            'RMSE': self.calculate_rmse(y_test, y_pred),
            'RMSD': self.calculate_rmsd(y_test, y_pred),
            'MAE': self.calculate_mae(y_test, y_pred),
            'Accuracy': self.calculate_accuracy(y_test, y_pred),
            'Precision': self.calculate_precision(y_test, y_pred),
            'AUC': self.calculate_auc(y_test, y_proba),
            'Confusion_Matrix': self.generate_confusion_matrix(y_test, y_pred)
        }
        
        self.metrics_results[model_name] = metrics
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Generate comparison DataFrame of all evaluated models."""
        if not self.metrics_results:
            raise ValueError("No models evaluated yet. Call evaluate_model() first.")
        
        comparison_data = []
        for model_name, metrics in self.metrics_results.items():
            row = {
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'RMSD': metrics['RMSD'],
                'MAE': metrics['MAE'],
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'AUC': metrics['AUC']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False).reset_index(drop=True)
        
        return comparison_df
    
    def verify_targets(self, model_name: str, tolerance: float = 0.05) -> dict:
        """
        Verify if a model meets target performance metrics.
        
        Args:
            model_name (str): Name of the model to verify
            tolerance (float): Acceptable deviation from target (default: 0.05 = 5%)
            
        Returns:
            dict: Verification results for each target metric
        """
        if model_name not in self.metrics_results:
            raise ValueError(f"Model '{model_name}' not found. Call evaluate_model() first.")
        
        metrics = self.metrics_results[model_name]
        verification = {}
        
        for metric_name, target_value in TARGET_METRICS.items():
            actual_value = metrics.get(metric_name, 0)
            
            if metric_name == 'RMSE':
                passed = actual_value <= target_value * (1 + tolerance)
                deviation = ((actual_value - target_value) / target_value) * 100
            else:
                passed = actual_value >= target_value * (1 - tolerance)
                deviation = ((actual_value - target_value) / target_value) * 100
            
            verification[metric_name] = {
                'target': target_value,
                'actual': actual_value,
                'deviation': deviation,
                'passed': passed
            }
        
        return verification
    
    def display_cv_results(self) -> None:
        """Display cross-validation results summary table."""
        if not self.cv_results:
            print("No cross-validation results available.")
            return
        
        print("" + "="*80)
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print("="*80)
        
        cv_data = []
        for model_name, results in self.cv_results.items():
            row = {'Model': model_name}
            for metric, values in results.items():
                metric_short = metric.replace('_weighted', '').replace('_', ' ').title()
                row[f'{metric_short} (Mean)'] = f"{values['mean']:.4f}"
                row[f'{metric_short} (Std)'] = f"{values['std']:.4f}"
            cv_data.append(row)
        
        cv_df = pd.DataFrame(cv_data)
        print("")
        print(cv_df.to_string(index=False))
        print("" + "="*80)
    
    def display_metrics_summary(self, model_name: str = None) -> None:
        """Display comprehensive metrics summary for one or all models."""
        if model_name:
            models_to_display = {model_name: self.metrics_results.get(model_name)}
        else:
            models_to_display = self.metrics_results
        
        for name, metrics in models_to_display.items():
            if metrics is None:
                continue
                
            print("" + "="*60)
            print(f"METRICS SUMMARY: {name}")
            print("="*60)
            
            print(f"Error Metrics:")
            print(f"    RMSE:      {metrics['RMSE']:.4f}")
            print(f"    RMSD:      {metrics['RMSD']:.4f}")
            print(f"    MAE:       {metrics['MAE']:.4f}")
            
            print(f" Classification Metrics: ")
            print(f"    Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
            print(f"    Precision: {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
            print(f"    AUC:       {metrics['AUC']:.4f}")
            
            print("" + "="*60)
