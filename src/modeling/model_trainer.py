"""
Model Training Module for Electric Vehicle Sales Analysis

This module contains functions for training, evaluating, and saving
machine learning models for EV sales prediction.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple, List, Optional

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



def load_processed_data(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data and separate features and target.
    
    Args:
        file_path: Path to the processed data file
        target_col: Name of the target column
        
    Returns:
        Tuple of (features_df, target_series)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the data")
    
    X = df.drop(columns=[target_col])
    # Drop 'Date' column if present
    if 'Date' in X.columns:
        X = X.drop(columns=['Date'])
    y = df[target_col]
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train multiple regression models on the training data.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dictionary of trained models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(tree_method='gpu_hist', random_state=42),
        'LightGBM': lgb.LGBMRegressor(device='gpu', random_state=42)
    }
    
    # Train each model
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models


def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trained models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics for each model
    """
    evaluation_results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        evaluation_results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    return evaluation_results


def tune_best_model(X_train: pd.DataFrame, y_train: pd.Series, best_model_name: str) -> Any:
    """
    Perform hyperparameter tuning for the best model.
    
    Args:
        X_train: Training features
        y_train: Training target
        best_model_name: Name of the best performing model
        
    Returns:
        Tuned model
    """
    # Define parameter grids for different models
    param_grids = {
        'Linear Regression': {},  # No hyperparameters to tune
        'Ridge Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Lasso Regression': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    # Select model and parameter grid
    if best_model_name == 'Linear Regression':
        model = LinearRegression()
    elif best_model_name == 'Ridge Regression':
        model = Ridge()
    elif best_model_name == 'Lasso Regression':
        model = Lasso()
    elif best_model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
    elif best_model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=42)
    elif best_model_name == 'XGBoost':
        model = xgb.XGBRegressor(tree_method='gpu_hist', random_state=42)
    elif best_model_name == 'LightGBM':
        model = lgb.LGBMRegressor(device='gpu', random_state=42)
    else:
        raise ValueError(f"Unknown model name: {best_model_name}")
    
    param_grid = param_grids[best_model_name]
    
    # If there are no parameters to tune, return the fitted model
    if not param_grid:
        model.fit(X_train, y_train)
        return model
    
    # Perform grid search
    print(f"Tuning hyperparameters for {best_model_name}...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_


def save_model(model: Any, model_path: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        model_path: Path where the model should be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model successfully saved to {model_path}")


def plot_feature_importance(model: Any, feature_names: List[str], output_path: str) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance visualization")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")


def main():
    """
    Main function to demonstrate the model training pipeline.
    """
    try:
        # Define paths
        data_path = "data/raw/ev_sales_data.csv"  # Use raw data file
        pipeline_path = "models/best_pipeline.pkl"
        target_col = "EV_Sales_Quantity"

        # Load raw data
        print(f"Loading raw data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded data with {df.shape[0]} samples and {df.shape[1]} features")

        # Clean data (reuse your clean_data function if available)
        from src.data_processing.data_processor import clean_data
        df = clean_data(df)

        # Drop 'Date' if present for modeling
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])

        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the data")
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split data
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Specify categorical columns explicitly
        categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
        print("Categorical columns:", categorical_cols)
        print("Numeric columns:", numeric_cols)

        # Build preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )

        # Choose your best model (e.g., XGBoost)
        model = xgb.XGBRegressor(tree_method='gpu_hist', random_state=42)

        # Build pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Fit pipeline
        print("Fitting pipeline (preprocessor + model)...")
        pipeline.fit(X_train, y_train)

        # Evaluate pipeline
        print("Evaluating pipeline...")
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Pipeline - RMSE: {rmse:.4f}, R2: {r2:.4f}")

        # Save pipeline
        print(f"Saving pipeline to {pipeline_path}...")
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Pipeline successfully saved to {pipeline_path}")

        print("Model training with pipeline completed successfully!")

    except Exception as e:
        print(f"Error in model training: {str(e)}")


if __name__ == "__main__":
    main()