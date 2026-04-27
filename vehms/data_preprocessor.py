# ============================================================
# DataPreprocessor Component
# ============================================================
# Purpose: Transform data for machine learning model training
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from .config import RANDOM_SEED


class DataPreprocessor:
    """
    DataPreprocessor class for transforming data for ML model training.
    
    This class provides methods to:
    - Separate features (X) from target variable (y)
    - Apply StandardScaler normalization to numeric features
    - Apply LabelEncoder to categorical target variable
    - Split data into training and testing sets with stratified sampling
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = RANDOM_SEED):
        """
        Initialize the DataPreprocessor instance.
        
        Args:
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
    
    def separate_features_target(self, df: pd.DataFrame, target: str = 'Decision') -> tuple:
        """
        Separate features (X) from target variable (y).
        
        Args:
            df (pd.DataFrame): DataFrame containing features and target
            target (str): Name of the target column (default: 'Decision')
            
        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series
        """
        X = df.drop(columns=[target])
        y = df[target]
        
        self.feature_names = X.columns.tolist()
        
        print("="*60)
        print("FEATURE-TARGET SEPARATION")
        print("="*60)
        print(f"\n[OK] Features (X): {X.shape[1]} columns")
        print(f"   Columns: {', '.join(self.feature_names)}")
        print(f"\n[OK] Target (y): '{target}' column")
        print(f"   Unique values: {y.unique().tolist()}")
        
        return X, y
    
    def fit_transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply StandardScaler normalization to numeric features.
        
        Args:
            X (pd.DataFrame): Features DataFrame to fit and transform
            
        Returns:
            np.ndarray: Scaled feature array
        """
        print("\n" + "="*60)
        print("FEATURE SCALING (StandardScaler)")
        print("="*60)
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\n[OK] StandardScaler fitted and applied to {X.shape[1]} features")
        print(f"\n[OK] All features now have mean ~ 0 and std ~ 1")
        
        return X_scaled
    
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted scaler.
        
        Args:
            X (pd.DataFrame): New features DataFrame to transform
            
        Returns:
            np.ndarray: Scaled feature array
        """
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler has not been fitted. Call fit_transform_features() first.")
        
        return self.scaler.transform(X)
    
    def encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Apply LabelEncoder to transform categorical target to numeric labels.
        
        Args:
            y (pd.Series): Target Series with categorical values
            
        Returns:
            np.ndarray: Encoded target array
        """
        print("\n" + "="*60)
        print("TARGET ENCODING (LabelEncoder)")
        print("="*60)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n[OK] LabelEncoder fitted and applied to target variable")
        print(f"\n   Label Encoding Mapping:")
        print("-" * 40)
        for idx, label in enumerate(self.label_encoder.classes_):
            count = (y_encoded == idx).sum()
            pct = (count / len(y_encoded)) * 100
            print(f"   {label:<10} -> {idx}  ({count:,} samples, {pct:.1f}%)")
        
        return y_encoded
    
    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Decode numeric labels back to original categories.
        
        Args:
            y_encoded (np.ndarray): Encoded target array
            
        Returns:
            np.ndarray: Decoded target array with original category names
        """
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("LabelEncoder has not been fitted. Call encode_target() first.")
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Split data into training and testing sets with stratified sampling.
        
        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Target array
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("TRAIN-TEST SPLIT (Stratified Sampling)")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"\n[OK] Data split with stratified sampling")
        print(f"   Random state: {self.random_state}")
        print(f"   Test size: {self.test_size * 100:.0f}%")
        
        print(f"\n   Training Set:")
        print(f"   - X_train shape: {X_train.shape}")
        print(f"   - y_train shape: {y_train.shape}")
        
        print(f"\n   Testing Set:")
        print(f"   - X_test shape: {X_test.shape}")
        print(f"   - y_test shape: {y_test.shape}")
        
        print(f"\n   Class Distribution Verification:")
        print("-" * 50)
        print(f"   {'Class':<10} {'Train':>10} {'Test':>10} {'Total':>10}")
        print("-" * 50)
        
        for idx, label in enumerate(self.label_encoder.classes_):
            train_count = (y_train == idx).sum()
            test_count = (y_test == idx).sum()
            total_count = train_count + test_count
            train_pct = (train_count / len(y_train)) * 100
            test_pct = (test_count / len(y_test)) * 100
            print(f"   {label:<10} {train_count:>6} ({train_pct:>4.1f}%) {test_count:>5} ({test_pct:>4.1f}%) {total_count:>6}")
        
        print("-" * 50)
        
        return X_train, X_test, y_train, y_test
