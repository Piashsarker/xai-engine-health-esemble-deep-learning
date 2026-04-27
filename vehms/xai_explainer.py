# ============================================================
# XAIExplainer Component
# ============================================================
# Purpose: Generate model explanations using SHAP and LIME
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular


class XAIExplainer:
    """
    XAIExplainer class for generating model explanations using SHAP and LIME.
    
    This class provides comprehensive explainability methods including:
    - SHAP KernelExplainer for ensemble models
    - SHAP summary, force, waterfall, and dependence plots
    - LIME tabular explanations for individual predictions
    - Combined analysis comparing SHAP and LIME insights
    """
    
    def __init__(self, model, feature_names: list, class_names: list):
        """
        Initialize the XAIExplainer instance.
        
        Args:
            model: Trained model to explain
            feature_names (list): Names of input features
            class_names (list): Names of target classes
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        self.X_sample = None
    
    # ============================================================
    # SHAP Methods
    # ============================================================
    
    def initialize_shap(self, X_background: np.ndarray, max_samples: int = 100) -> None:
        """
        Initialize SHAP explainer using KernelExplainer for stacked models.
        
        Args:
            X_background (np.ndarray): Background data for SHAP
            max_samples (int): Maximum samples for background (default: 100)
        """
        print("\n[...] Initializing SHAP explainer...")
        
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_bg = X_background[indices]
        else:
            X_bg = X_background
        
        # Convert to DataFrame for better SHAP compatibility
        if isinstance(X_bg, np.ndarray):
            X_bg_df = pd.DataFrame(X_bg, columns=self.feature_names)
        else:
            X_bg_df = X_bg
        
        self.shap_explainer = shap.KernelExplainer(
            self.model.predict_proba, 
            X_bg_df
        )
        
        print(f"   [OK] SHAP KernelExplainer initialized")
        print(f"   Background samples: {len(X_bg)}")
    
    def compute_shap_values(self, X_sample: np.ndarray, nsamples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for a sample of data.
        
        Args:
            X_sample (np.ndarray): Sample data to explain
            nsamples (int): Number of samples for SHAP approximation
            
        Returns:
            np.ndarray: SHAP values for each sample and feature
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap() first.")
        
        print(f"\n[...] Computing SHAP values for {len(X_sample)} samples...")
        print(f"   This may take a few minutes...")
        
        # Store X_sample as DataFrame with feature names for SHAP compatibility
        if isinstance(X_sample, np.ndarray):
            self.X_sample = pd.DataFrame(X_sample, columns=self.feature_names)
        else:
            self.X_sample = X_sample
            
        self.shap_values = self.shap_explainer.shap_values(X_sample, nsamples=nsamples)
        
        # Debug: print shapes to understand the structure
        if isinstance(self.shap_values, list):
            print(f"   SHAP values: list of {len(self.shap_values)} arrays")
            print(f"   Each array shape: {self.shap_values[0].shape}")
        else:
            print(f"   SHAP values shape: {self.shap_values.shape}")
        print(f"   X_sample shape: {self.X_sample.shape}")
        
        print(f"   [OK] SHAP values computed successfully!")
        
        return self.shap_values
    
    def plot_shap_summary(self, class_idx: int = None) -> None:
        """
        Generate SHAP summary plot showing feature importance.
        
        Args:
            class_idx (int, optional): Class index to plot. If None, plots for all classes.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        print("\n" + "="*60)
        print("SHAP SUMMARY PLOT")
        print("="*60)
        
        # Get the SHAP values for the specified class
        if isinstance(self.shap_values, list):
            # Multi-class: shap_values is a list of arrays [n_classes][n_samples, n_features]
            if class_idx is not None:
                shap_vals = self.shap_values[class_idx]
                title = f"SHAP Summary - Class: {self.class_names[class_idx]}"
            else:
                # Compute mean absolute values across all classes
                shap_vals = np.abs(np.array(self.shap_values)).mean(axis=0)
                title = "SHAP Summary - Mean Absolute Values"
        else:
            # Single output or already stacked
            shap_vals = self.shap_values
            title = "SHAP Summary"
        
        # Ensure shapes match
        n_samples = self.X_sample.shape[0]
        n_features = self.X_sample.shape[1]
        
        print(f"   SHAP values shape: {shap_vals.shape}")
        print(f"   X_sample shape: {self.X_sample.shape}")
        
        # Verify shapes match
        if shap_vals.shape[0] != n_samples or shap_vals.shape[1] != n_features:
            print(f"   [!] Shape mismatch detected, attempting to fix...")
            # Try transposing if dimensions are swapped
            if shap_vals.shape[0] == n_features and shap_vals.shape[1] == n_samples:
                shap_vals = shap_vals.T
                print(f"   [OK] Transposed SHAP values to shape: {shap_vals.shape}")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, self.X_sample, show=False)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] SHAP summary plot generated successfully")
    
    def plot_shap_bar(self, class_idx: int = 0) -> None:
        """
        Generate SHAP bar plot showing mean absolute feature importance.
        
        Args:
            class_idx (int): Class index to plot (default: 0)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        print("\n" + "="*60)
        print("SHAP BAR PLOT")
        print("="*60)
        
        # Get SHAP values for the specified class
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        # Ensure shapes match
        n_samples = self.X_sample.shape[0]
        n_features = self.X_sample.shape[1]
        
        if shap_vals.shape[0] != n_samples or shap_vals.shape[1] != n_features:
            if shap_vals.shape[0] == n_features and shap_vals.shape[1] == n_samples:
                shap_vals = shap_vals.T
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, self.X_sample, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - Class: {self.class_names[class_idx]}", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] SHAP bar plot generated successfully")
    
    def plot_shap_force(self, sample_idx: int = 0, class_idx: int = 0) -> None:
        """
        Generate SHAP force plot for a single prediction.
        
        Args:
            sample_idx (int): Index of sample to explain (default: 0)
            class_idx (int): Class index to plot (default: 0)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        print("\n" + "="*60)
        print(f"SHAP FORCE PLOT - Sample {sample_idx}, Class: {self.class_names[class_idx]}")
        print("="*60)
        
        shap.initjs()
        
        force_plot = shap.force_plot(
            self.shap_explainer.expected_value[class_idx],
            self.shap_values[class_idx][sample_idx],
            self.X_sample[sample_idx],
            feature_names=self.feature_names
        )
        
        return force_plot
    
    def plot_shap_dependence(self, feature: str, class_idx: int = 0) -> None:
        """
        Generate SHAP dependence plot for a specific feature.
        
        Args:
            feature (str): Feature name to plot
            class_idx (int): Class index to plot (default: 0)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        print("\n" + "="*60)
        print(f"SHAP DEPENDENCE PLOT - {feature}")
        print("="*60)
        
        feature_idx = self.feature_names.index(feature)
        
        # Get SHAP values for the specified class
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        # Ensure shapes match
        n_samples = self.X_sample.shape[0]
        n_features = self.X_sample.shape[1]
        
        if shap_vals.shape[0] != n_samples or shap_vals.shape[1] != n_features:
            if shap_vals.shape[0] == n_features and shap_vals.shape[1] == n_samples:
                shap_vals = shap_vals.T
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, 
            shap_vals, 
            self.X_sample,
            show=False
        )
        plt.title(f"SHAP Dependence - {feature} (Class: {self.class_names[class_idx]})",
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] SHAP dependence plot generated successfully")
    
    # ============================================================
    # LIME Methods
    # ============================================================
    
    def initialize_lime(self, X_train: np.ndarray, mode: str = 'classification') -> None:
        """
        Initialize LIME explainer for tabular data.
        
        Args:
            X_train (np.ndarray): Training data for LIME
            mode (str): Explanation mode ('classification' or 'regression')
        """
        print("\n[...] Initializing LIME explainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            discretize_continuous=True
        )
        
        print(f"   [OK] LIME TabularExplainer initialized")
        print(f"   Mode: {mode}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Classes: {self.class_names}")
    
    def explain_instance_lime(self, instance: np.ndarray, num_features: int = 9,
                               top_labels: int = 4) -> object:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance (np.ndarray): Single instance to explain
            num_features (int): Number of features to include in explanation
            top_labels (int): Number of top labels to explain
            
        Returns:
            lime.explanation.Explanation: LIME explanation object
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime() first.")
        
        print("\n[...] Generating LIME explanation...")
        
        explanation = self.lime_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features,
            top_labels=top_labels
        )
        
        print("   [OK] LIME explanation generated successfully")
        
        return explanation
    
    def plot_lime_explanation(self, explanation, label: int = None) -> None:
        """
        Plot LIME explanation for a specific label.
        
        Args:
            explanation: LIME explanation object
            label (int, optional): Label to plot. If None, plots top label.
        """
        print("\n" + "="*60)
        print("LIME EXPLANATION PLOT")
        print("="*60)
        
        if label is None:
            label = explanation.available_labels()[0]
        
        fig = explanation.as_pyplot_figure(label=label)
        plt.title(f"LIME Explanation - Class: {self.class_names[label]}", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] LIME explanation plot generated successfully")
    
    def display_lime_text(self, explanation, label: int = None) -> None:
        """
        Display LIME explanation as text.
        
        Args:
            explanation: LIME explanation object
            label (int, optional): Label to display. If None, displays top label.
        """
        if label is None:
            label = explanation.available_labels()[0]
        
        print("\n" + "="*60)
        print(f"LIME EXPLANATION - Class: {self.class_names[label]}")
        print("="*60)
        
        print(f"\nPredicted probabilities:")
        probs = explanation.predict_proba
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}: {probs[i]:.4f}")
        
        print(f"\nFeature contributions for {self.class_names[label]}:")
        print("-" * 50)
        
        exp_list = explanation.as_list(label=label)
        for feature, weight in exp_list:
            direction = "+" if weight > 0 else ""
            print(f"   {feature:<30} : {direction}{weight:.4f}")
        
        print("-" * 50)
    
    # ============================================================
    # Combined Analysis
    # ============================================================
    
    def compare_shap_lime(self, instance: np.ndarray, sample_idx: int = 0,
                          class_idx: int = 0) -> pd.DataFrame:
        """
        Compare SHAP and LIME feature importance for a single instance.
        
        Args:
            instance (np.ndarray): Single instance to explain
            sample_idx (int): Index of sample in SHAP values
            class_idx (int): Class index to compare
            
        Returns:
            pd.DataFrame: Comparison of SHAP and LIME importance
        """
        print("\n" + "="*60)
        print("SHAP vs LIME COMPARISON")
        print("="*60)
        
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime() first.")
        
        shap_importance = np.abs(self.shap_values[class_idx][sample_idx])
        
        lime_exp = self.explain_instance_lime(instance)
        lime_dict = dict(lime_exp.as_list(label=class_idx))
        
        comparison_data = []
        for i, feature in enumerate(self.feature_names):
            lime_val = 0
            for key, val in lime_dict.items():
                if feature.lower() in key.lower():
                    lime_val = abs(val)
                    break
            
            comparison_data.append({
                'Feature': feature,
                'SHAP Importance': shap_importance[i],
                'LIME Importance': lime_val
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('SHAP Importance', ascending=False)
        
        print(f"\nClass: {self.class_names[class_idx]}")
        print("-" * 60)
        print(comparison_df.to_string(index=False))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        ax1.barh(comparison_df['Feature'], comparison_df['SHAP Importance'], color='steelblue')
        ax1.set_xlabel('Absolute SHAP Value')
        ax1.set_title('SHAP Feature Importance', fontweight='bold')
        ax1.invert_yaxis()
        
        ax2 = axes[1]
        ax2.barh(comparison_df['Feature'], comparison_df['LIME Importance'], color='coral')
        ax2.set_xlabel('Absolute LIME Weight')
        ax2.set_title('LIME Feature Importance', fontweight='bold')
        ax2.invert_yaxis()
        
        plt.suptitle(f'SHAP vs LIME Comparison - Class: {self.class_names[class_idx]}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n[OK] SHAP vs LIME comparison complete")
        
        return comparison_df
