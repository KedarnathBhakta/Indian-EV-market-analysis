"""
Visualization Module for Electric Vehicle Sales Analysis

This module contains functions for creating various visualizations
to analyze and present insights from the EV sales data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Optional, List, Dict, Any


def set_plotting_style():
    """Set consistent style for matplotlib plots."""
    plt.style.use('seaborn-whitegrid')
    sns.set_palette('viridis')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def save_plot(fig, filename: str, output_dir: str = "../../reports/figures"):
    """
    Save a matplotlib figure to the specified output directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the output file
        output_dir: Directory to save the figure
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def plot_sales_by_state(df: pd.DataFrame, state_col: str = 'State', sales_col: str = 'Sales', 
                        title: str = 'Electric Vehicle Sales by State', 
                        filename: str = 'sales_by_state.png'):
    """
    Create a bar plot of EV sales by state.
    
    Args:
        df: DataFrame containing the data
        state_col: Name of the column containing state information
        sales_col: Name of the column containing sales information
        title: Title for the plot
        filename: Name of the output file
    """
    # Sort by sales for better visualization
    plot_data = df.groupby(state_col)[sales_col].sum().sort_values(ascending=False).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x=state_col, y=sales_col, data=plot_data, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('State', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # Add data labels
    for i, v in enumerate(plot_data[sales_col]):
        ax.text(i, v + 0.1, f"{v:,.0f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, filename)


def plot_sales_trend(df: pd.DataFrame, date_col: str = 'Date', sales_col: str = 'Sales',
                    group_by: str = 'M', title: str = 'Electric Vehicle Sales Trend',
                    filename: str = 'sales_trend.png'):
    """
    Create a line plot of EV sales trend over time.
    
    Args:
        df: DataFrame containing the data
        date_col: Name of the column containing date information
        sales_col: Name of the column containing sales information
        group_by: Time period to group by ('D' for day, 'W' for week, 'M' for month, 'Q' for quarter, 'Y' for year)
        title: Title for the plot
        filename: Name of the output file
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by time period
    time_series = df.groupby(pd.Grouper(key=date_col, freq=group_by))[sales_col].sum().reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(time_series[date_col], time_series[sales_col], marker='o', linestyle='-', linewidth=2)
    
    # Add trend line
    z = np.polyfit(range(len(time_series)), time_series[sales_col], 1)
    p = np.poly1d(z)
    ax.plot(time_series[date_col], p(range(len(time_series))), "r--", linewidth=1)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, filename)


def plot_sales_distribution(df: pd.DataFrame, sales_col: str = 'Sales',
                           title: str = 'Distribution of Electric Vehicle Sales',
                           filename: str = 'sales_distribution.png'):
    """
    Create a histogram and KDE plot of the sales distribution.
    
    Args:
        df: DataFrame containing the data
        sales_col: Name of the column containing sales information
        title: Title for the plot
        filename: Name of the output file
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(df[sales_col], kde=True, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Sales', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Add descriptive statistics
    stats = f"Mean: {df[sales_col].mean():.2f}\nMedian: {df[sales_col].median():.2f}\nStd Dev: {df[sales_col].std():.2f}"
    ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_plot(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, title: str = 'Correlation Matrix',
                            filename: str = 'correlation_heatmap.png'):
    """
    Create a heatmap of the correlation matrix for numeric columns.
    
    Args:
        df: DataFrame containing the data
        title: Title for the plot
        filename: Name of the output file
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)
    
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    save_plot(fig, filename)


def plot_sales_by_category(df: pd.DataFrame, category_col: str, sales_col: str = 'Sales',
                          title: Optional[str] = None, filename: Optional[str] = None):
    """
    Create a bar plot of EV sales by a categorical variable.
    
    Args:
        df: DataFrame containing the data
        category_col: Name of the column containing category information
        sales_col: Name of the column containing sales information
        title: Title for the plot (default: 'Electric Vehicle Sales by {category_col}')
        filename: Name of the output file (default: 'sales_by_{category_col.lower()}.png')
    """
    if title is None:
        title = f'Electric Vehicle Sales by {category_col}'
    
    if filename is None:
        filename = f'sales_by_{category_col.lower()}.png'
    
    # Aggregate data by category
    plot_data = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=category_col, y=sales_col, data=plot_data, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(category_col, fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot(fig, filename)


def create_interactive_map(df: pd.DataFrame, state_col: str = 'State', sales_col: str = 'Sales',
                          title: str = 'Electric Vehicle Sales by State',
                          filename: str = 'sales_map.html'):
    """
    Create an interactive choropleth map of EV sales by state using Plotly.
    
    Args:
        df: DataFrame containing the data
        state_col: Name of the column containing state information
        sales_col: Name of the column containing sales information
        title: Title for the plot
        filename: Name of the output file
    """
    try:
        # Aggregate data by state
        state_sales = df.groupby(state_col)[sales_col].sum().reset_index()
        
        # Create choropleth map
        fig = px.choropleth(
            state_sales,
            locations=state_col,
            locationmode='country names',  # For Indian states, you might need a custom approach
            color=sales_col,
            hover_name=state_col,
            color_continuous_scale='Viridis',
            title=title
        )
        
        # Update layout
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'
            ),
            coloraxis_colorbar=dict(
                title='Sales'
            )
        )
        
        # Save the interactive plot
        output_dir = "../../reports/figures"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path)
        print(f"Interactive map saved to {output_path}")
        
    except Exception as e:
        print(f"Error creating interactive map: {str(e)}")
        print("Note: For Indian states, you might need a custom GeoJSON file or a different approach.")


def plot_model_comparison(model_metrics: Dict[str, Dict[str, float]], metric: str = 'R2',
                         title: Optional[str] = None, filename: str = 'model_comparison.png'):
    """
    Create a bar plot comparing different models based on a specific metric.
    
    Args:
        model_metrics: Dictionary of model metrics (output from model evaluation)
        metric: Metric to use for comparison ('R2', 'RMSE', 'MAE', etc.)
        title: Title for the plot (default: '{metric} Comparison of Models')
        filename: Name of the output file
    """
    if title is None:
        title = f'{metric} Comparison of Models'
    
    # Extract metric values for each model
    models = list(model_metrics.keys())
    values = [metrics[metric] for metrics in model_metrics.values()]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(models, values)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot(fig, filename)


def main():
    """
    Main function to demonstrate the visualization capabilities.
    """
    try:
        # Set plotting style
        set_plotting_style()
        
        # Load data
        data_path = "../../data/processed/processed_ev_sales_data.csv"  # Update with your actual file path
        print(f"Loading data from {data_path}...")
        
        try:
            df = pd.read_csv(data_path)
            print(f"Loaded data with shape: {df.shape}")
            
            # Example visualizations
            # Note: These will only work if your data has the expected columns
            # Adjust column names as needed
            
            # Check if required columns exist
            required_cols = ['State', 'Sales', 'Date']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: The following required columns are missing: {missing_cols}")
                print("Some visualizations may not be created.")
            
            # Create visualizations based on available columns
            if 'State' in df.columns and 'Sales' in df.columns:
                print("Creating sales by state visualization...")
                plot_sales_by_state(df)
            
            if 'Date' in df.columns and 'Sales' in df.columns:
                print("Creating sales trend visualization...")
                plot_sales_trend(df)
            
            if 'Sales' in df.columns:
                print("Creating sales distribution visualization...")
                plot_sales_distribution(df)
            
            print("Creating correlation heatmap...")
            plot_correlation_heatmap(df)
            
            # Example of creating visualizations for categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'State' and col != 'Date':  # Skip already visualized columns
                    print(f"Creating sales by {col} visualization...")
                    plot_sales_by_category(df, col)
            
            print("Visualization process completed successfully!")
            
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            print("Please ensure the processed data file exists.")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")


if __name__ == "__main__":
    main()
    """
Visualization Module for Electric Vehicle Sales Analysis

This module contains functions for creating various visualizations
to analyze and present insights from the EV sales data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Optional, List, Dict, Any


def set_plotting_style():
    """Set consistent style for matplotlib plots."""
    plt.style.use('seaborn-whitegrid')
    sns.set_palette('viridis')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def save_plot(fig, filename: str, output_dir: str = "../../reports/figures"):
    """
    Save a matplotlib figure to the specified output directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the output file
        output_dir: Directory to save the figure
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def plot_sales_by_state(df: pd.DataFrame, state_col: str = 'State', sales_col: str = 'Sales', 
                        title: str = 'Electric Vehicle Sales by State', 
                        filename: str = 'sales_by_state.png'):
    """
    Create a bar plot of EV sales by state.
    
    Args:
        df: DataFrame containing the data
        state_col: Name of the column containing state information
        sales_col: Name of the column containing sales information
        title: Title for the plot
        filename: Name of the output file
    """
    # Sort by sales for better visualization
    plot_data = df.groupby(state_col)[sales_col].sum().sort_values(ascending=False).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x=state_col, y=sales_col, data=plot_data, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('State', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # Add data labels
    for i, v in enumerate(plot_data[sales_col]):
        ax.text(i, v + 0.1, f"{v:,.0f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, filename)


def plot_sales_trend(df: pd.DataFrame, date_col: str = 'Date', sales_col: str = 'Sales',
                    group_by: str = 'M', title: str = 'Electric Vehicle Sales Trend',
                    filename: str = 'sales_trend.png'):
    """
    Create a line plot of EV sales trend over time.
    
    Args:
        df: DataFrame containing the data
        date_col: Name of the column containing date information
        sales_col: Name of the column containing sales information
        group_by: Time period to group by ('D' for day, 'W' for week, 'M' for month, 'Q' for quarter, 'Y' for year)
        title: Title for the plot
        filename: Name of the output file
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by time period
    time_series = df.groupby(pd.Grouper(key=date_col, freq=group_by))[sales_col].sum().reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(time_series[date_col], time_series[sales_col], marker='o', linestyle='-', linewidth=2)
    
    # Add trend line
    z = np.polyfit(range(len(time_series)), time_series[sales_col], 1)
    p = np.poly1d(z)
    ax.plot(time_series[date_col], p(range(len(time_series))), "r--", linewidth=1)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, filename)


def plot_sales_distribution(df: pd.DataFrame, sales_col: str = 'Sales',
                           title: str = 'Distribution of Electric Vehicle Sales',
                           filename: str = 'sales_distribution.png'):
    """
    Create a histogram and KDE plot of the sales distribution.
    
    Args:
        df: DataFrame containing the data
        sales_col: Name of the column containing sales information
        title: Title for the plot
        filename: Name of the output file
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(df[sales_col], kde=True, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Sales', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Add descriptive statistics
    stats = f"Mean: {df[sales_col].mean():.2f}\nMedian: {df[sales_col].median():.2f}\nStd Dev: {df[sales_col].std():.2f}"
    ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_plot(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, title: str = 'Correlation Matrix',
                            filename: str = 'correlation_heatmap.png'):
    """
    Create a heatmap of the correlation matrix for numeric columns.
    
    Args:
        df: DataFrame containing the data
        title: Title for the plot
        filename: Name of the output file
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)
    
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    save_plot(fig, filename)


def plot_sales_by_category(df: pd.DataFrame, category_col: str, sales_col: str = 'Sales',
                          title: Optional[str] = None, filename: Optional[str] = None):
    """
    Create a bar plot of EV sales by a categorical variable.
    
    Args:
        df: DataFrame containing the data
        category_col: Name of the column containing category information
        sales_col: Name of the column containing sales information
        title: Title for the plot (default: 'Electric Vehicle Sales by {category_col}')
        filename: Name of the output file (default: 'sales_by_{category_col.lower()}.png')
    """
    if title is None:
        title = f'Electric Vehicle Sales by {category_col}'
    
    if filename is None:
        filename = f'sales_by_{category_col.lower()}.png'
    
    # Aggregate data by category
    plot_data = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=category_col, y=sales_col, data=plot_data, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(category_col, fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot(fig, filename)


def create_interactive_map(df: pd.DataFrame, state_col: str = 'State', sales_col: str = 'Sales',
                          title: str = 'Electric Vehicle Sales by State',
                          filename: str = 'sales_map.html'):
    """
    Create an interactive choropleth map of EV sales by state using Plotly.
    
    Args:
        df: DataFrame containing the data
        state_col: Name of the column containing state information
        sales_col: Name of the column containing sales information
        title: Title for the plot
        filename: Name of the output file
    """
    try:
        # Aggregate data by state
        state_sales = df.groupby(state_col)[sales_col].sum().reset_index()
        
        # Create choropleth map
        fig = px.choropleth(
            state_sales,
            locations=state_col,
            locationmode='country names',  # For Indian states, you might need a custom approach
            color=sales_col,
            hover_name=state_col,
            color_continuous_scale='Viridis',
            title=title
        )
        
        # Update layout
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'
            ),
            coloraxis_colorbar=dict(
                title='Sales'
            )
        )
        
        # Save the interactive plot
        output_dir = "../../reports/figures"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path)
        print(f"Interactive map saved to {output_path}")
        
    except Exception as e:
        print(f"Error creating interactive map: {str(e)}")
        print("Note: For Indian states, you might need a custom GeoJSON file or a different approach.")


def plot_model_comparison(model_metrics: Dict[str, Dict[str, float]], metric: str = 'R2',
                         title: Optional[str] = None, filename: str = 'model_comparison.png'):
    """
    Create a bar plot comparing different models based on a specific metric.
    
    Args:
        model_metrics: Dictionary of model metrics (output from model evaluation)
        metric: Metric to use for comparison ('R2', 'RMSE', 'MAE', etc.)
        title: Title for the plot (default: '{metric} Comparison of Models')
        filename: Name of the output file
    """
    if title is None:
        title = f'{metric} Comparison of Models'
    
    # Extract metric values for each model
    models = list(model_metrics.keys())
    values = [metrics[metric] for metrics in model_metrics.values()]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(models, values)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot(fig, filename)


def main():
    """
    Main function to demonstrate the visualization capabilities.
    """
    try:
        # Set plotting style
        set_plotting_style()
        
        # Load data
        data_path = "../../data/processed/processed_ev_sales_data.csv"  # Update with your actual file path
        print(f"Loading data from {data_path}...")
        
        try:
            df = pd.read_csv(data_path)
            print(f"Loaded data with shape: {df.shape}")
            
            # Example visualizations
            # Note: These will only work if your data has the expected columns
            # Adjust column names as needed
            
            # Check if required columns exist
            required_cols = ['State', 'Sales', 'Date']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: The following required columns are missing: {missing_cols}")
                print("Some visualizations may not be created.")
            
            # Create visualizations based on available columns
            if 'State' in df.columns and 'Sales' in df.columns:
                print("Creating sales by state visualization...")
                plot_sales_by_state(df)
            
            if 'Date' in df.columns and 'Sales' in df.columns:
                print("Creating sales trend visualization...")
                plot_sales_trend(df)
            
            if 'Sales' in df.columns:
                print("Creating sales distribution visualization...")
                plot_sales_distribution(df)
            
            print("Creating correlation heatmap...")
            plot_correlation_heatmap(df)
            
            # Example of creating visualizations for categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'State' and col != 'Date':  # Skip already visualized columns
                    print(f"Creating sales by {col} visualization...")
                    plot_sales_by_category(df, col)
            
            print("Visualization process completed successfully!")
            
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            print("Please ensure the processed data file exists.")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")


if __name__ == "__main__":
    main()