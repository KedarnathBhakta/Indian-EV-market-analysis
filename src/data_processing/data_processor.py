"""
Data Processing Module for Electric Vehicle Sales Analysis

This module contains functions for loading, cleaning, and preprocessing
the electric vehicle sales data.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from the specified file path.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded DataFrame
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by handling missing values,
    removing duplicates, and correcting data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values (example strategies)
    # For numeric columns: fill with median
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # For categorical columns: fill with mode
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown")
    
    # Convert date columns if any (assuming a column named 'Date' exists)
    if 'Date' in cleaned_df.columns:
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'], errors='coerce')
    
    return cleaned_df


def preprocess_data(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Preprocess the data for analysis or modeling.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column for modeling (if applicable)
        
    Returns:
        Tuple of (features_df, target_series) where target_series is None if target_col is None
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Extract target if specified
    target = None
    if target_col and target_col in processed_df.columns:
        target = processed_df[target_col]
        processed_df = processed_df.drop(columns=[target_col])
    
    # One-hot encode categorical variables
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    
    return processed_df, target


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed DataFrame to the specified output path.
    
    Args:
        df: DataFrame to save
        output_path: Path where the DataFrame should be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save based on file extension
    file_extension = os.path.splitext(output_path)[1].lower()
    
    if file_extension == '.csv':
        df.to_csv(output_path, index=False)
    elif file_extension in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format for saving: {file_extension}")
    
    print(f"Data successfully saved to {output_path}")


def main():
    """
    Main function to demonstrate the data processing pipeline.
    """
    # Example usage
    try:
        # Define paths
        input_path = "../../data/raw/ev_sales_data.csv"  # Update with your actual file path
        output_path = "../../data/processed/processed_ev_sales_data.csv"
        
        # Load data
        print(f"Loading data from {input_path}...")
        df = load_data(input_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Clean data
        print("Cleaning data...")
        cleaned_df = clean_data(df)
        print(f"Cleaned data shape: {cleaned_df.shape}")
        
        # Preprocess data
        print("Preprocessing data...")
        processed_df, target = preprocess_data(cleaned_df, target_col="EV_Sales_Quantity")
        if target is not None:
            processed_df["EV_Sales_Quantity"] = target
        # Drop the 'Date' column if it exists
        if 'Date' in processed_df.columns:
            processed_df = processed_df.drop(columns=['Date'])
        print(f"Processed data shape: {processed_df.shape}")
        
        # Save processed data
        print(f"Saving processed data to {output_path}...")
        save_processed_data(processed_df, output_path)
        
        print("Data processing completed successfully!")
        
    except Exception as e:
        print(f"Error in data processing: {str(e)}")


if __name__ == "__main__":
    main()