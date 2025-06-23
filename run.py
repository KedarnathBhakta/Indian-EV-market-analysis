"""
Main script to run the Electric Vehicle Sales Analysis project.

This script provides a command-line interface to run different components
of the project, such as data processing, model training, and visualization.
"""

import argparse
import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def process_data():
    """Run the data processing pipeline."""
    print("Running data processing pipeline...")
    from src.data_processing.data_processor import load_data, clean_data, preprocess_data, save_processed_data
    
    # Check if raw data exists
    raw_data_path = os.path.join("data", "raw", "ev_sales_data.csv")
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please place your data file in the data/raw directory.")
        return False
    
    # Load and process data
    try:
        print(f"Loading data from {raw_data_path}...")
        df = load_data(raw_data_path)
        print(f"Loaded data with shape: {df.shape}")
        
        print("Cleaning data...")
        cleaned_df = clean_data(df)
        print(f"Cleaned data shape: {cleaned_df.shape}")
        
        print("Preprocessing data...")
        processed_df, _ = preprocess_data(cleaned_df)
        print(f"Processed data shape: {processed_df.shape}")
        
        # Save processed data
        output_path = os.path.join("data", "processed", "processed_ev_sales_data.csv")
        print(f"Saving processed data to {output_path}...")
        save_processed_data(processed_df, output_path)
        
        print("Data processing completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        return False


def train_model():
    """Run the model training pipeline."""
    print("Running model training pipeline...")
    
    # Check if processed data exists
    processed_data_path = os.path.join("data", "processed", "processed_ev_sales_data.csv")
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please run data processing first.")
        return False
    
    # Train model
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        from src.modeling.model_trainer import (
            load_processed_data, split_data, train_models, 
            evaluate_models, tune_best_model, save_model
        )
        
        # Define target column
        target_col = "EV_Sales_Quantity"
        
        # Load data
        print(f"Loading processed data from {processed_data_path}...")
        X, y = load_processed_data(processed_data_path, target_col)
        print(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")
        
        # --- Create a proper preprocessing pipeline ---
        # Identify numeric columns (that are not one-hot encoded)
        numeric_features = [col for col in X.columns if X[col].nunique() > 2]
        # One-hot encoded columns will be passed through without scaling
        passthrough_features = [col for col in X.columns if X[col].nunique() <= 2]

        print(f"Identified {len(numeric_features)} numeric features to scale.")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('passthrough', 'passthrough', passthrough_features)
            ],
            remainder='passthrough' # Ensure no columns are accidentally dropped
        )

        # Split data before fitting the preprocessor
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Fit the preprocessor on the training data and transform both sets
        print("Fitting preprocessor and transforming data...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save the preprocessor
        preprocessor_path = os.path.join("models", "preprocessor.pkl")
        save_model(preprocessor, preprocessor_path) # Reusing save_model for the preprocessor
        print(f"Preprocessor saved to {preprocessor_path}")
        
        # Train models on the correctly processed data
        print("Training models on processed data...")
        models = train_models(X_train_processed, y_train)
        
        # Evaluate models
        print("Evaluating models...")
        evaluation_results = evaluate_models(models, X_test_processed, y_test)
        
        # Find best model based on R2 score
        best_model_name = max(evaluation_results.items(), key=lambda x: x[1]['R2'])[0]
        print(f"Best model: {best_model_name}")
        
        # Tune best model
        print("Tuning best model...")
        best_model = tune_best_model(X_train_processed, y_train, best_model_name)
        
        # Save best model
        model_path = os.path.join("models", "best_model.pkl")
        print(f"Saving best model to {model_path}...")
        save_model(best_model, model_path)
        
        print("Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        return False


def create_visualizations():
    """Create visualizations from the processed data."""
    print("Creating visualizations...")
    
    # Check if processed data exists
    processed_data_path = os.path.join("data", "processed", "processed_ev_sales_data.csv")
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please run data processing first.")
        return False
    
    # Create visualizations
    try:
        from src.visualization.visualizer import (
            set_plotting_style, plot_sales_by_state, plot_sales_trend,
            plot_sales_distribution, plot_correlation_heatmap
        )
        
        # Set plotting style
        set_plotting_style()
        
        # Load data
        print(f"Loading data from {processed_data_path}...")
        df = pd.read_csv(processed_data_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Create visualizations based on available columns
        if 'State' in df.columns and 'EV_Sales_Quantity' in df.columns:
            print("Creating sales by state visualization...")
            plot_sales_by_state(df)
        
        if 'Date' in df.columns and 'EV_Sales_Quantity' in df.columns:
            print("Creating sales trend visualization...")
            plot_sales_trend(df)
        
        if 'EV_Sales_Quantity' in df.columns:
            print("Creating sales distribution visualization...")
            plot_sales_distribution(df)
        
        print("Creating correlation heatmap...")
        plot_correlation_heatmap(df)
        
        print("Visualization creation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in creating visualizations: {str(e)}")
        return False


def run_app():
    """Run the Streamlit app."""
    print("Starting Streamlit app...")
    
    try:
        # Check if Streamlit is installed
        subprocess.run(["streamlit", "--version"], check=True, capture_output=True)
        
        # Run the app
        app_path = os.path.join("app", "app.py")
        subprocess.run(["streamlit", "run", app_path], check=True)
        
        return True
        
    except subprocess.CalledProcessError:
        print("Error: Streamlit is not installed. Please install it with 'pip install streamlit'.")
        return False
    except Exception as e:
        print(f"Error running Streamlit app: {str(e)}")
        return False


def main():
    """Main function to parse arguments and run the appropriate component."""
    parser = argparse.ArgumentParser(description="Electric Vehicle Sales Analysis")
    parser.add_argument("--process", action="store_true", help="Run data processing pipeline")
    parser.add_argument("--train", action="store_true", help="Train machine learning models")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--app", action="store_true", help="Run Streamlit app")
    parser.add_argument("--all", action="store_true", help="Run all components")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run components based on arguments
    if args.all or args.process:
        process_data()
    
    if args.all or args.train:
        train_model()
    
    if args.all or args.visualize:
        create_visualizations()
    
    if args.all or args.app:
        run_app()


if __name__ == "__main__":
    main()