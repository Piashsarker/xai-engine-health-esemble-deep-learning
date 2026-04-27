# ============================================================
# DataLoader Component
# ============================================================
# Purpose: Load and validate the engine sensor dataset
# ============================================================

import pandas as pd
import numpy as np
from .config import REQUIRED_COLUMNS, VALID_DECISIONS


class DataLoader:
    """
    DataLoader class for loading and validating the VEHMS engine sensor dataset.
    
    This class provides methods to:
    - Load CSV files into pandas DataFrames
    - Validate that all required columns exist
    - Verify data types for numeric and categorical columns
    - Display dataset summary statistics
    """
    
    def __init__(self):
        """Initialize the DataLoader instance."""
        self.df = None
        self.validation_errors = []
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.
        
        Args:
            filepath (str): Path to the CSV file to load
            
        Returns:
            pd.DataFrame: The loaded dataset
        """
        try:
            self.df = pd.read_csv(filepath)
            print(f"[OK] Dataset loaded successfully from: {filepath}")
            print(f"  Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")
    
    def validate_columns(self, df: pd.DataFrame = None) -> bool:
        """
        Validate that all required columns exist in the dataset.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to validate. Uses self.df if not provided.
            
        Returns:
            bool: True if all required columns exist, False otherwise
        """
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        existing_columns = set(df.columns)
        required_columns = set(REQUIRED_COLUMNS)
        missing_columns = required_columns - existing_columns
        
        if missing_columns:
            error_msg = f"Missing required columns: {sorted(missing_columns)}"
            self.validation_errors.append(error_msg)
            raise ValueError(error_msg)
        
        print(f"[OK] All {len(REQUIRED_COLUMNS)} required columns validated")
        return True
    
    def validate_data_types(self, df: pd.DataFrame = None) -> bool:
        """
        Validate data types for numeric and categorical columns.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to validate. Uses self.df if not provided.
            
        Returns:
            bool: True if all data types are valid, False otherwise
        """
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        numeric_columns = [col for col in REQUIRED_COLUMNS if col != 'Decision']
        
        print("\n[OK] Validating numeric columns...")
        invalid_numeric = []
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.isna().sum() > df[col].isna().sum():
                        invalid_numeric.append(col)
        
        if invalid_numeric:
            error_msg = f"Invalid numeric values in columns: {invalid_numeric}"
            self.validation_errors.append(error_msg)
            raise ValueError(error_msg)
        
        print(f"  All {len(numeric_columns)} numeric columns contain valid float values")
        
        print("\n[OK] Validating Decision column categories...")
        if 'Decision' in df.columns:
            unique_decisions = set(df['Decision'].dropna().unique())
            valid_decisions = set(VALID_DECISIONS)
            invalid_decisions = unique_decisions - valid_decisions
            
            if invalid_decisions:
                error_msg = f"Invalid Decision values found: {sorted(invalid_decisions)}"
                self.validation_errors.append(error_msg)
                raise ValueError(error_msg)
            
            print(f"  Valid categories found: {sorted(unique_decisions)}")
            
            print("\n  Category Distribution:")
            for category in VALID_DECISIONS:
                count = (df['Decision'] == category).sum()
                percentage = (count / len(df)) * 100
                print(f"    {category}: {count} samples ({percentage:.1f}%)")
        
        return True
    
    def display_summary(self, df: pd.DataFrame = None) -> None:
        """
        Display comprehensive dataset summary including shape, types, and statistics.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to summarize. Uses self.df if not provided.
        """
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        print(f"\n[STATS] Dataset Shape:")
        print(f"   Rows (samples): {df.shape[0]:,}")
        print(f"   Columns (features): {df.shape[1]}")
        
        print(f"\n[LIST] Column Data Types:")
        print("-" * 40)
        for col in df.columns:
            dtype = df[col].dtype
            print(f"   {col:<15} : {dtype}")
        
        print(f"\n[CHART] Numeric Column Statistics:")
        print("-" * 60)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe().T
        stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        print(stats_df.to_string())
        
        print(f"\n[SEARCH] Missing Values:")
        print("-" * 40)
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   No missing values detected")
        else:
            for col in df.columns:
                if missing[col] > 0:
                    print(f"   {col}: {missing[col]} missing values")
        
        print("\n" + "="*60)
