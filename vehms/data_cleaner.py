# ============================================================
# DataCleaner Component
# ============================================================
# Purpose: Handle missing values, duplicates, and outliers
# ============================================================

import pandas as pd
import numpy as np


class DataCleaner:
    """
    DataCleaner class for handling data quality issues in the VEHMS dataset.
    
    This class provides methods to:
    - Identify and count missing values per column
    - Detect and remove duplicate rows
    - Impute missing values using statistical methods
    - Handle outliers using IQR or z-score methods
    """
    
    def __init__(self, outlier_method: str = 'iqr'):
        """
        Initialize the DataCleaner instance.
        
        Args:
            outlier_method (str): Method for outlier detection ('iqr' or 'zscore')
        """
        self.outlier_method = outlier_method
        self.cleaning_report = {
            'missing_values': {},
            'duplicates_removed': 0,
            'values_imputed': {},
            'outliers_handled': {},
            'original_shape': None,
            'final_shape': None
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.Series:
        """Identify and count missing values per column."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        self.cleaning_report['missing_values'] = {
            col: {'count': int(missing[col]), 'percentage': float(missing_pct[col])}
            for col in df.columns
        }
        
        print("="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        
        total_missing = missing.sum()
        if total_missing == 0:
            print("\n[OK] No missing values detected in the dataset!")
        else:
            print(f"\n[!] Total missing values: {total_missing}")
            print("\nMissing values per column:")
            for col in df.columns:
                if missing[col] > 0:
                    print(f"   {col:<15}: {missing[col]:>5} ({missing_pct[col]:>5.2f}%)")
        
        return missing
    
    def check_duplicates(self, df: pd.DataFrame) -> int:
        """Identify duplicate rows in the dataset."""
        duplicates = df.duplicated().sum()
        
        print("\n" + "="*60)
        print("DUPLICATE ROWS ANALYSIS")
        print("="*60)
        
        if duplicates == 0:
            print("\n[OK] No duplicate rows detected in the dataset!")
        else:
            print(f"\n[!] Duplicate rows found: {duplicates}")
            print(f"   Percentage of dataset: {(duplicates / len(df)) * 100:.2f}%")
        
        return duplicates
    
    def impute_missing(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Impute missing values using specified strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with missing values
            strategy (str): Imputation strategy ('median', 'mean', 'mode')
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        df_imputed = df.copy()
        imputed_info = {}
        
        print("\n" + "="*60)
        print(f"MISSING VALUE IMPUTATION (Strategy: {strategy.upper()})")
        print("="*60)
        
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in numeric_cols:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                if strategy == 'median':
                    fill_value = df_imputed[col].median()
                elif strategy == 'mean':
                    fill_value = df_imputed[col].mean()
                else:
                    fill_value = df_imputed[col].mode()[0]
                
                df_imputed[col].fillna(fill_value, inplace=True)
                imputed_info[col] = {'count': int(missing_count), 'strategy': strategy, 'fill_value': float(fill_value)}
                print(f"\n[OK] {col}: Imputed {missing_count} values with {strategy} = {fill_value:.4f}")
        
        for col in categorical_cols:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                fill_value = df_imputed[col].mode()[0]
                df_imputed[col].fillna(fill_value, inplace=True)
                imputed_info[col] = {'count': int(missing_count), 'strategy': 'mode', 'fill_value': str(fill_value)}
                print(f"\n[OK] {col}: Imputed {missing_count} values with mode = {fill_value}")
        
        self.cleaning_report['values_imputed'] = imputed_info
        
        if not imputed_info:
            print("\n[OK] No imputation needed - no missing values found!")
        
        return df_imputed
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        original_count = len(df)
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        removed_count = original_count - len(df_cleaned)
        
        self.cleaning_report['duplicates_removed'] = removed_count
        
        print("\n" + "="*60)
        print("DUPLICATE REMOVAL")
        print("="*60)
        
        if removed_count == 0:
            print("\n[OK] No duplicates to remove!")
        else:
            print(f"\n[OK] Removed {removed_count} duplicate rows")
            print(f"   Original rows: {original_count:,}")
            print(f"   Remaining rows: {len(df_cleaned):,}")
        
        return df_cleaned
    
    def handle_outliers(self, df: pd.DataFrame, columns: list = None,
                        method: str = None, action: str = 'clip') -> pd.DataFrame:
        """
        Detect and handle outliers using IQR or z-score method.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            columns (list, optional): Columns to check for outliers
            method (str, optional): Outlier detection method ('iqr' or 'zscore')
            action (str): How to handle outliers ('clip', 'remove', 'flag')
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        df_cleaned = df.copy()
        method = method or self.outlier_method
        
        if columns is None:
            columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        print("\n" + "="*60)
        print(f"OUTLIER HANDLING (Method: {method.upper()}, Action: {action.upper()})")
        print("="*60)
        
        outlier_info = {}
        total_outliers_handled = 0
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            else:
                mean = df_cleaned[col].mean()
                std = df_cleaned[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
            
            outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': int(outlier_count),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'action': action
                }
                
                if action == 'clip':
                    df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                    print(f"\n[OK] {col}: Clipped {outlier_count} outliers")
                elif action == 'flag':
                    print(f"\n[!] {col}: {outlier_count} outliers detected (not modified)")
                
                total_outliers_handled += outlier_count
        
        self.cleaning_report['outliers_handled'] = outlier_info
        
        if total_outliers_handled == 0:
            print("\n[OK] No outliers detected in the specified columns!")
        else:
            print(f"\n" + "-"*40)
            print(f"Total outliers handled: {total_outliers_handled}")
        
        return df_cleaned
    
    def get_cleaning_report(self) -> dict:
        """Return summary of all cleaning operations performed."""
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY REPORT")
        print("="*60)
        
        if self.cleaning_report['original_shape']:
            print(f"\n[STATS] Dataset Shape:")
            print(f"   Original: {self.cleaning_report['original_shape'][0]:,} rows x {self.cleaning_report['original_shape'][1]} columns")
            if self.cleaning_report['final_shape']:
                print(f"   Final:    {self.cleaning_report['final_shape'][0]:,} rows x {self.cleaning_report['final_shape'][1]} columns")
        
        print(f"\n[FILE] Duplicates Removed: {self.cleaning_report['duplicates_removed']}")
        
        print("\n" + "="*60)
        print("[OK] DATA CLEANING COMPLETE")
        print("="*60)
        
        return self.cleaning_report
