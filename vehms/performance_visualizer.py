# ============================================================
# PerformanceVisualizer Component
# ============================================================
# Purpose: Generate model performance visualizations
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


class PerformanceVisualizer:
    """
    PerformanceVisualizer class for generating model performance visualizations.
    
    This class provides methods to:
    - Generate model comparison bar charts for various metrics
    - Create ROC curves for all models on the same plot
    - Generate confusion matrix heatmaps
    - Create comprehensive comparison dashboards
    """
    
    def __init__(self, figsize: tuple = (14, 10)):
        """
        Initialize the PerformanceVisualizer instance.
        
        Args:
            figsize (tuple): Default figure size for plots (width, height)
        """
        self.figsize = figsize
        self.colors = {
            'KNN': '#1f77b4',
            'SVM': '#ff7f0e',
            'Random Forest': '#2ca02c',
            'AdaBoost': '#d62728',
            'XGBoost': '#9467bd',
            'Stacked Model 1': '#8c564b',
            'Stacked Model 2': '#e377c2',
            'Stacked Model 3': '#7f7f7f',
            'ER-Stacked Model 1': '#3498db',
            'ER-Stacked Model 2': '#2980b9',
            'ER-Stacked Model 3': '#1a5276',
            'DS-Stack Performance': '#2ecc71',
            'DS-Stack Diversity': '#27ae60',
            'DS-Stack Combined': '#1e8449',
        }
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                               metrics: list = None,
                               highlight_best: bool = True) -> None:
        """Generate bar charts comparing model performance across metrics."""
        if metrics is None:
            metrics = ['Accuracy', 'AUC', 'Precision', 'RMSE']
        
        print("="*80)
        print("MODEL PERFORMANCE COMPARISON CHARTS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        models = comparison_df['Model'].tolist()
        colors = [self.colors.get(m, '#333333') for m in models]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = comparison_df[metric].values
            
            bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.5)
            
            if highlight_best:
                if metric == 'RMSE':
                    best_idx = np.argmin(values)
                else:
                    best_idx = np.argmax(values)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.tick_params(axis='x', rotation=45)
            
            target_metrics = {'Accuracy': 0.9470, 'AUC': 0.9702, 'Precision': 0.9486, 'RMSE': 0.3355}
            if metric in target_metrics:
                ax.axhline(y=target_metrics[metric], color='red', linestyle='--', 
                          linewidth=2, label=f'Target: {target_metrics[metric]}')
                ax.legend(loc='lower right')
            
            if metric != 'RMSE':
                ax.set_ylim([min(values) * 0.95, 1.0])
            else:
                ax.set_ylim([0, max(values) * 1.15])
        
        plt.suptitle('Model Performance Comparison Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] Model comparison charts generated successfully")
    
    def plot_roc_curves(self, models: dict, X_test: np.ndarray, y_test: np.ndarray,
                        class_names: list = None) -> None:
        """Generate ROC curves for all models on the same plot."""
        print("\n" + "="*80)
        print("ROC CURVES - ALL MODELS")
        print("="*80)
        
        n_classes = len(np.unique(y_test))
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)
            else:
                y_score = model.decision_function(X_test)
                y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
            
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            
            macro_auc = np.mean(list(roc_auc.values()))
            
            color = self.colors.get(model_name, '#333333')
            linewidth = 3 if 'Stacked Model 1' in model_name else 2
            ax.plot(all_fpr, mean_tpr, color=color, linewidth=linewidth,
                   label=f'{model_name} (AUC = {macro_auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-Class (Macro-Average)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] ROC curves generated successfully")
    
    def plot_confusion_matrices(self, models: dict, X_test: np.ndarray, 
                                 y_test: np.ndarray, class_names: list) -> None:
        """Generate confusion matrix heatmaps for all models."""
        print("\n" + "="*80)
        print("CONFUSION MATRIX HEATMAPS")
        print("="*80)
        
        n_models = len(models)
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else axes
        
        for idx, (model_name, model) in enumerate(models.items()):
            ax = axes[idx]
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            annot = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar=True, square=True,
                       annot_kws={'size': 9})
            
            accuracy = accuracy_score(y_test, y_pred)
            
            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.4f}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
        
        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] Confusion matrix heatmaps generated successfully")
    
    def plot_metrics_radar(self, comparison_df: pd.DataFrame, 
                           models_to_plot: list = None) -> None:
        """Generate radar chart comparing models across multiple metrics."""
        print("\n" + "="*80)
        print("RADAR CHART - MODEL COMPARISON")
        print("="*80)
        
        if models_to_plot is None:
            models_to_plot = comparison_df['Model'].tolist()
        
        metrics = ['Accuracy', 'Precision', 'AUC']
        
        df_filtered = comparison_df[comparison_df['Model'].isin(models_to_plot)]
        
        fig = go.Figure()
        
        for _, row in df_filtered.iterrows():
            model_name = row['Model']
            values = [row[m] for m in metrics]
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.9, 1.0]
                )
            ),
            showlegend=True,
            title=dict(
                text='Model Performance Radar Chart',
                font=dict(size=16)
            ),
            width=800,
            height=600
        )
        
        fig.show()
        
        print("\n[OK] Radar chart generated successfully")
    
    def create_comparison_dashboard(self, comparison_df: pd.DataFrame,
                                     models: dict, X_test: np.ndarray,
                                     y_test: np.ndarray, class_names: list) -> None:
        """Create comprehensive comparison dashboard with all visualizations."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON DASHBOARD")
        print("="*80)
        
        self.plot_model_comparison(comparison_df)
        self.plot_roc_curves(models, X_test, y_test, class_names)
        self.plot_confusion_matrices(models, X_test, y_test, class_names)
        self.plot_metrics_radar(comparison_df)
        
        print("\n" + "="*80)
        print("[OK] COMPREHENSIVE DASHBOARD COMPLETE")
        print("="*80)
