# ============================================================
# DataVisualizer Component
# ============================================================
# Purpose: Generate exploratory data analysis visualizations
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class DataVisualizer:
    """
    DataVisualizer class for generating exploratory data analysis visualizations.
    
    This class provides methods to:
    - Generate histogram distributions for numeric features
    - Create correlation heatmaps showing feature relationships
    - Display class distribution for target variables
    - Produce box plots for outlier detection
    - Create pairwise scatter plots for feature relationships
    """
    
    def __init__(self, figsize: tuple = (12, 8), color_palette: str = 'husl'):
        """
        Initialize the DataVisualizer instance.
        
        Args:
            figsize (tuple): Default figure size for plots (width, height)
            color_palette (str): Seaborn color palette name
        """
        self.figsize = figsize
        self.color_palette = color_palette
        sns.set_palette(color_palette)
    
    def plot_distributions(self, df: pd.DataFrame, columns: list = None, 
                           bins: int = 30, kde: bool = True) -> None:
        """Generate histogram distributions for numeric features."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        print("="*60)
        print("FEATURE DISTRIBUTION PLOTS")
        print("="*60)
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            sns.histplot(data=df, x=col, bins=bins, kde=kde, ax=ax, 
                        color=sns.color_palette(self.color_palette, n_colors=len(columns))[idx])
            mean_val = df[col].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Distribution of Numeric Features', fontsize=14, fontweight='bold', y=1.02)
        plt.show()
        
        print("[OK] Distribution plots generated successfully")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, columns: list = None,
                                  method: str = 'pearson', annot: bool = True,
                                  cmap: str = 'RdYlBu_r') -> None:
        """Generate correlation matrix heatmap showing relationships between features."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("="*60)
        print("CORRELATION HEATMAP")
        print("="*60)
        
        corr_matrix = df[columns].corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=annot, cmap=cmap,
                    vmin=-1, vmax=1, center=0, square=True,
                    linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                    fmt='.2f', annot_kws={'size': 9}, ax=ax)
        
        ax.set_title(f'Feature Correlation Heatmap ({method.capitalize()} Method)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] Correlation heatmap generated successfully")
    
    def plot_class_distribution(self, df: pd.DataFrame, target: str = 'Decision',
                                 colors: list = None) -> None:
        """Generate bar chart for target class distribution."""
        print("="*60)
        print("TARGET CLASS DISTRIBUTION")
        print("="*60)
        
        class_counts = df[target].value_counts()
        class_percentages = df[target].value_counts(normalize=True) * 100
        
        if colors is None:
            colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        bars = ax1.bar(class_counts.index, class_counts.values, color=colors, 
                       edgecolor='black', linewidth=1.2)
        
        for bar, count, pct in zip(bars, class_counts.values, class_percentages.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        ax1.set_title(f'{target} Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Engine Health Status', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_ylim(0, max(class_counts.values) * 1.15)
        
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index,
                                            colors=colors, autopct='%1.1f%%',
                                            startangle=90, explode=[0.02]*len(class_counts),
                                            shadow=True, textprops={'fontsize': 11})
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        ax2.set_title(f'{target} Class Proportions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\n[STATS] Class Distribution Summary:")
        print("-" * 40)
        for cls in class_counts.index:
            count = class_counts[cls]
            pct = class_percentages[cls]
            print(f"   {cls:<10}: {count:>5,} samples ({pct:>5.1f}%)")
        
        print("\n[OK] Class distribution plots generated successfully")
    
    def plot_boxplots(self, df: pd.DataFrame, columns: list = None,
                       hue: str = None, orient: str = 'v') -> None:
        """Generate box plots for outlier detection in numeric features."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("="*60)
        print("BOX PLOTS FOR OUTLIER DETECTION")
        print("="*60)
        
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        outlier_stats = {}
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            
            if hue:
                sns.boxplot(data=df, x=hue, y=col, ax=ax, palette=self.color_palette)
            else:
                sns.boxplot(data=df, y=col, ax=ax, 
                           color=sns.color_palette(self.color_palette, n_colors=len(columns))[idx])
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_stats[col] = len(outliers)
            
            ax.set_title(f'{col}\n(Outliers: {len(outliers)})', fontsize=11, fontweight='bold')
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Box Plots for Outlier Detection', fontsize=14, fontweight='bold', y=1.02)
        plt.show()
        
        print("\n[STATS] Outlier Summary (IQR Method):")
        print("-" * 40)
        for col, count in sorted(outlier_stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(df)) * 100
            print(f"   {col:<15}: {count:>5} outliers ({pct:>5.2f}%)")
        
        print("\n[OK] Box plots generated successfully")
    
    def plot_pairplot(self, df: pd.DataFrame, columns: list = None,
                       hue: str = 'Decision', diag_kind: str = 'kde',
                       corner: bool = False) -> None:
        """Generate pairwise scatter plots for feature relationships."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("="*60)
        print("PAIRWISE SCATTER PLOTS")
        print("="*60)
        print(f"\nGenerating pairplot for {len(columns)} features...")
        print("(This may take a moment for large datasets)\n")
        
        plot_columns = columns.copy()
        if hue and hue in df.columns and hue not in plot_columns:
            plot_data = df[plot_columns + [hue]]
        else:
            plot_data = df[plot_columns]
            hue = None
        
        palette = {'Good': '#2ecc71', 'Minor': '#f39c12', 
                   'Moderate': '#e74c3c', 'Critical': '#9b59b6'}
        
        g = sns.pairplot(plot_data, hue=hue, diag_kind=diag_kind,
                         corner=corner, palette=palette if hue else None,
                         plot_kws={'alpha': 0.6, 's': 20},
                         diag_kws={'alpha': 0.7})
        
        g.fig.suptitle('Pairwise Feature Relationships by Engine Health Status',
                       fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        # Create interactive plotly scatter matrix for subset of features
        print("\n[STATS] Interactive Scatter Matrix (Top 5 Features):")
        
        # Select top 5 features for interactive plot (to keep it manageable)
        top_features = columns[:5] if len(columns) > 5 else columns
        
        fig = px.scatter_matrix(
            df,
            dimensions=top_features,
            color=hue if hue and hue in df.columns else None,
            color_discrete_map=palette if hue else None,
            title='Interactive Scatter Matrix',
            opacity=0.6
        )
        
        fig.update_layout(
            title_font_size=16,
            width=900,
            height=900
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.show()
        
        print("\n[OK] Pairwise scatter plots generated successfully")
    
    def plot_interactive_class_distribution(self, df: pd.DataFrame, target: str = 'Decision',
                                             colors: list = None) -> None:
        """Generate interactive plotly bar chart for target class distribution."""
        class_counts = df[target].value_counts()
        
        if colors is None:
            colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        
        fig = px.bar(x=class_counts.index, y=class_counts.values,
                     color=class_counts.index,
                     color_discrete_sequence=colors,
                     title=f'Interactive {target} Distribution',
                     labels={'x': 'Engine Health Status', 'y': 'Count'})
        
        fig.update_layout(
            showlegend=False,
            title_font_size=16,
            xaxis_title_font_size=12,
            yaxis_title_font_size=12
        )
        
        fig.show()
    
    def plot_grouped_boxplots(self, df: pd.DataFrame, columns: list = None,
                               target: str = 'Decision') -> None:
        """Generate grouped box plots by target class."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("\n[STATS] Box Plots Grouped by Engine Health Status:")
        
        # Melt dataframe for grouped visualization
        df_melted = df.melt(id_vars=[target], value_vars=columns,
                            var_name='Feature', value_name='Value')
        
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.boxplot(data=df_melted, x='Feature', y='Value', hue=target,
                   palette=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'], ax=ax)
        
        ax.set_title('Feature Distributions by Engine Health Status', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(title='Decision', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
