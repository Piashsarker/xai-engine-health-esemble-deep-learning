# ============================================================
# FeatureExtractor Component
# ============================================================
# Purpose: Calculate and analyze feature importance
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from .config import RANDOM_SEED


class FeatureExtractor:
    """
    FeatureExtractor class for calculating and analyzing feature importance.
    
    This class provides methods to:
    - Calculate feature importance scores using tree-based methods (RandomForest)
    - Rank features by their importance to the prediction task
    - Generate feature importance visualizations (bar charts)
    - Identify top contributing features for engine health prediction
    """
    
    def __init__(self):
        """Initialize the FeatureExtractor instance."""
        self.importance_scores = None
        self.feature_names = None
        self.model = None
    
    def calculate_importance(self, X: np.ndarray, y: np.ndarray, 
                             feature_names: list = None,
                             method: str = 'random_forest',
                             random_state: int = RANDOM_SEED) -> pd.DataFrame:
        """
        Calculate feature importance using tree-based methods.
        
        Args:
            X (np.ndarray): Feature array (training data)
            y (np.ndarray): Target array (training labels)
            feature_names (list, optional): List of feature names for labeling
            method (str): Method for importance calculation (default: 'random_forest')
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: DataFrame with feature names and importance scores
        """
        print("="*60)
        print("FEATURE IMPORTANCE CALCULATION")
        print("="*60)
        print(f"\nMethod: {method.replace('_', ' ').title()}")
        print(f"Training samples: {X.shape[0]:,}")
        print(f"Features: {X.shape[1]}")
        
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        print(f"\n[...] Training RandomForestClassifier for importance calculation...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        importance_values = self.model.feature_importances_
        
        self.importance_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance_values
        })
        
        print(f"\n[OK] Feature importance calculated successfully!")
        print(f"\n   Model Training Accuracy: {self.model.score(X, y):.4f}")
        
        return self.importance_scores
    
    def rank_features(self) -> pd.DataFrame:
        """
        Rank features by their importance score in descending order.
        
        Returns:
            pd.DataFrame: DataFrame with features sorted by importance (highest first)
        """
        if self.importance_scores is None:
            raise ValueError("Importance scores not calculated. Call calculate_importance() first.")
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE RANKING")
        print("="*60)
        
        ranked_features = self.importance_scores.sort_values(
            by='Importance', 
            ascending=False
        ).reset_index(drop=True)
        
        ranked_features.insert(0, 'Rank', range(1, len(ranked_features) + 1))
        
        total_importance = ranked_features['Importance'].sum()
        ranked_features['Percentage'] = (ranked_features['Importance'] / total_importance * 100).round(2)
        ranked_features['Cumulative %'] = ranked_features['Percentage'].cumsum().round(2)
        
        print(f"\n[STATS] Ranked Feature Importance Table:")
        print("-" * 70)
        print(f"{'Rank':<6} {'Feature':<15} {'Importance':>12} {'Percentage':>12} {'Cumulative':>12}")
        print("-" * 70)
        
        for _, row in ranked_features.iterrows():
            print(f"{int(row['Rank']):<6} {row['Feature']:<15} {row['Importance']:>12.6f} {row['Percentage']:>11.2f}% {row['Cumulative %']:>11.2f}%")
        
        print("-" * 70)
        
        self.importance_scores = ranked_features
        
        return ranked_features
    
    def plot_importance(self, figsize: tuple = (12, 8), 
                        color_palette: str = 'viridis',
                        top_n: int = None) -> None:
        """
        Generate a horizontal bar chart showing feature importance.
        
        Args:
            figsize (tuple): Figure size (width, height)
            color_palette (str): Color palette for the bars
            top_n (int, optional): Number of top features to display
        """
        if self.importance_scores is None:
            raise ValueError("Importance scores not calculated. Call calculate_importance() first.")
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE VISUALIZATION")
        print("="*60)
        
        if 'Rank' in self.importance_scores.columns:
            plot_data = self.importance_scores.copy()
        else:
            plot_data = self.importance_scores.sort_values('Importance', ascending=False).copy()
        
        if top_n is not None:
            plot_data = plot_data.head(top_n)
            title_suffix = f" (Top {top_n})"
        else:
            title_suffix = ""
        
        plot_data = plot_data.sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        cmap = plt.colormaps.get_cmap(color_palette)
        colors = cmap(np.linspace(0.3, 0.9, len(plot_data)))
        
        bars = ax.barh(plot_data['Feature'], plot_data['Importance'], 
                       color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, importance in zip(bars, plot_data['Importance']):
            width = bar.get_width()
            ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', ha='left', va='center',
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Importance for Engine Health Prediction{title_suffix}',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0, plot_data['Importance'].max() * 1.15)
        
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] Feature importance visualization generated successfully!")
    
    def get_top_features(self, n: int = 5) -> list:
        """
        Return the top n most important features.
        
        Args:
            n (int): Number of top features to return (default: 5)
            
        Returns:
            list: List of top n feature names sorted by importance
        """
        if self.importance_scores is None:
            raise ValueError("Importance scores not calculated. Call calculate_importance() first.")
        
        print("\n" + "="*60)
        print(f"TOP {n} CONTRIBUTING FEATURES")
        print("="*60)
        
        if 'Rank' in self.importance_scores.columns:
            top_features_df = self.importance_scores.head(n)
        else:
            top_features_df = self.importance_scores.nlargest(n, 'Importance')
        
        top_features = top_features_df['Feature'].tolist()
        
        print(f"\n[TROPHY] Top {n} Features for Engine Health Prediction:")
        print("-" * 50)
        
        for i, (_, row) in enumerate(top_features_df.iterrows(), 1):
            importance = row['Importance']
            if 'Percentage' in row:
                pct = row['Percentage']
                print(f"   {i}. {row['Feature']:<15} (Importance: {importance:.4f}, {pct:.2f}%)")
            else:
                print(f"   {i}. {row['Feature']:<15} (Importance: {importance:.4f})")
        
        print("-" * 50)
        
        if 'Percentage' in top_features_df.columns:
            total_contribution = top_features_df['Percentage'].sum()
            print(f"\n   Combined contribution: {total_contribution:.2f}% of total importance")
        
        print(f"\n[OK] Top {n} features identified successfully!")
        
        return top_features
