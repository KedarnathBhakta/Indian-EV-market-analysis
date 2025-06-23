"""
Streamlit App for Electric Vehicle Sales Analysis

This app provides an interactive dashboard for exploring and visualizing
electric vehicle sales data across different states in India.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import sys
import pickle
from datetime import datetime
from sklearn.pipeline import Pipeline

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Electric Vehicle Sales Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Add project root to sys.path ---
# This ensures the app can find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import custom modules ---
from src.data_processing.data_processor import load_data, clean_data
from src.visualization.visualizer import set_plotting_style


@st.cache_data
def load_and_process_data(file_path):
    """Load and process data with caching for performance."""
    try:
        df = load_data(file_path)
        df = clean_data(df)
        
        # Handle data type issues for Streamlit compatibility
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Convert any problematic columns to string to avoid Arrow serialization issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_resource
def load_model(model_path):
    """Load the trained model with caching."""
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            st.warning(f"Model file not found at {model_path}")
            return None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Add some debugging information
        st.success(f"Model loaded successfully from {model_path}")
        st.write(f"Model type: {type(model).__name__}")
        if hasattr(model, 'feature_importances_'):
            st.write(f"Number of features: {len(model.feature_importances_)}")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write(f"Exception type: {type(e).__name__}")
        return None


@st.cache_resource
def load_preprocessor(preprocessor_path):
    """Load the fitted preprocessor with caching."""
    try:
        if not os.path.exists(preprocessor_path):
            st.warning(f"Preprocessor file not found at {preprocessor_path}")
            return None
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        return preprocessor
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return None


def main():
    """Main function to run the Streamlit app."""
    # Set plotting style
    set_plotting_style()

    # --- Robust Path Configuration ---
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, "data", "processed", "processed_ev_sales_data.csv")
    model_path = os.path.join(project_root, "models", "best_model.pkl")
    preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Overview", "Data Explorer", "Sales Analysis", "State Comparison", "Predictions"]
    )
    
    # --- Improved Data Loading Logic ---
    df = load_and_process_data(data_path)

    if df is None:
        st.error(f"Could not load the processed data file from: {data_path}")
        st.warning("Please ensure the processed data file exists or upload a raw data file below.")
        
        uploaded_file = st.file_uploader("Upload Raw CSV Data", type=['csv'])
        if uploaded_file is not None:
            # When a raw file is uploaded, it needs to be processed.
            raw_df = pd.read_csv(uploaded_file)
            st.success("Raw data uploaded successfully! Processing...")
            # Process the raw dataframe
            df = clean_data(raw_df) 
            # Also run the other processing steps
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            st.success("Data processed and ready for analysis.")
        else:
            # Keeps the app from breaking if no data is available
            st.stop() 
    
    # Display selected page
    if page == "Overview":
        display_overview(df)
    elif page == "Data Explorer":
        display_data_explorer(df)
    elif page == "Sales Analysis":
        display_sales_analysis(df)
    elif page == "State Comparison":
        display_state_comparison(df)
    elif page == "Predictions":
        display_predictions(df, model_path, preprocessor_path)


def display_overview(df):
    """Display overview page with key metrics and summary."""
    st.title("Electric Vehicle Sales in India")
    st.markdown("### Overview Dashboard")
    
    if df is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'EV_Sales_Quantity' in df.columns:
                total_sales = df['EV_Sales_Quantity'].sum()
                st.metric("Total Sales", f"{total_sales:,.0f}")
            else:
                st.metric("Total Sales", "N/A")
        
        with col2:
            # Count number of states (one-hot encoded columns starting with 'State_')
            state_cols = [col for col in df.columns if col.startswith('State_')]
            if state_cols:
                num_states = len(state_cols)
                st.metric("States", num_states)
            else:
                st.metric("States", "N/A")
        
        with col3:
            if 'Date' in df.columns:
                # Convert Date column to datetime first
                df_copy = df.copy()
                df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
                # Filter out any invalid dates
                valid_dates = df_copy['Date'].dropna()
                if not valid_dates.empty:
                    date_range = f"{valid_dates.min().strftime('%b %Y')} - {valid_dates.max().strftime('%b %Y')}"
                    st.metric("Time Period", date_range)
                else:
                    st.metric("Time Period", "N/A")
            else:
                st.metric("Time Period", "N/A")
        
        with col4:
            if 'EV_Sales_Quantity' in df.columns and 'Date' in df.columns:
                # Convert Date column to datetime first
                df_copy = df.copy()
                df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
                # Filter out any invalid dates
                valid_data = df_copy.dropna(subset=['Date', 'EV_Sales_Quantity'])
                if not valid_data.empty:
                    avg_monthly_sales = valid_data.groupby(valid_data['Date'].dt.to_period('M'))['EV_Sales_Quantity'].sum().mean()
                    st.metric("Avg. Monthly Sales", f"{avg_monthly_sales:,.0f}")
                else:
                    st.metric("Avg. Monthly Sales", "N/A")
            else:
                st.metric("Avg. Monthly Sales", "N/A")
        
        # Summary charts
        st.markdown("### Summary Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'EV_Sales_Quantity' in df.columns:
                st.subheader("Top 5 States by Sales")
                # Calculate total sales by state using one-hot encoded columns
                state_cols = [col for col in df.columns if col.startswith('State_')]
                if state_cols:
                    state_sales = {}
                    for state_col in state_cols:
                        state_name = state_col.replace('State_', '')
                        total_state_sales = df[df[state_col] == 1]['EV_Sales_Quantity'].sum()
                        state_sales[state_name] = total_state_sales
                    
                    # Get top 5 states
                    top_states = dict(sorted(state_sales.items(), key=lambda x: x[1], reverse=True)[:5])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    states = list(top_states.keys())
                    sales = list(top_states.values())
                    ax.bar(states, sales)
                    ax.set_ylabel('Sales')
                    ax.set_title('Top 5 States by EV Sales')
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
                else:
                    st.info("State data not available for visualization")
            else:
                st.info("Sales data not available for visualization")
        
        with col2:
            if 'Date' in df.columns and 'EV_Sales_Quantity' in df.columns:
                st.subheader("Sales Trend")
                # Convert Date column to datetime first
                df_copy = df.copy()
                df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
                # Filter out any invalid dates
                valid_data = df_copy.dropna(subset=['Date', 'EV_Sales_Quantity'])
                if not valid_data.empty:
                    monthly_sales = valid_data.groupby(valid_data['Date'].dt.to_period('M'))['EV_Sales_Quantity'].sum().reset_index()
                    monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(monthly_sales['Date'], monthly_sales['EV_Sales_Quantity'], marker='o')
                    ax.set_ylabel('Sales')
                    ax.set_title('Monthly EV Sales Trend')
                    st.pyplot(fig)
                else:
                    st.info("No valid date data available for visualization")
            else:
                st.info("Date or Sales data not available for visualization")
    else:
        st.info("Please upload data to view the overview dashboard.")


def display_data_explorer(df):
    """Display data explorer page with interactive data table and summary statistics."""
    st.title("Data Explorer")
    
    if df is not None:
        # Data summary
        st.subheader("Data Summary")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display column information
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)
        
        # Interactive data table
        st.subheader("Data Preview")
        
        # Filters
        st.markdown("#### Filters")
        filter_cols = st.multiselect("Select columns to filter by:", list(df.columns))
        
        filtered_df = df.copy()
        for col in filter_cols:
            if df[col].dtype == 'object':
                selected_values = st.multiselect(f"Select {col}:", list(df[col].unique()))
                if selected_values:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
            elif pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.slider(f"Select {col} range:", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]
        
        # Display filtered data
        st.dataframe(filtered_df)
        st.write(f"Showing {filtered_df.shape[0]} of {df.shape[0]} rows")
        
        # Summary statistics
        if st.checkbox("Show summary statistics"):
            st.subheader("Summary Statistics")
            st.dataframe(filtered_df.describe())
    else:
        st.info("Please upload data to explore.")


def display_sales_analysis(df):
    """Display sales analysis page with various visualizations."""
    st.title("Sales Analysis")
    
    if df is not None:
        # Time period selection
        if 'Date' in df.columns:
            # Convert Date column to datetime first
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
            # Filter out any invalid dates
            valid_data = df_copy.dropna(subset=['Date'])
            if not valid_data.empty:
                min_date, max_date = valid_data['Date'].min(), valid_data['Date'].max()
                # Convert to datetime objects for Streamlit slider
                min_date_dt = min_date.to_pydatetime()
                max_date_dt = max_date.to_pydatetime()
                selected_period = st.slider(
                    "Select time period:",
                    min_value=min_date_dt,
                    max_value=max_date_dt,
                    value=(min_date_dt, max_date_dt)
                )
                filtered_df = valid_data[(valid_data['Date'] >= selected_period[0]) & (valid_data['Date'] <= selected_period[1])]
            else:
                filtered_df = df
                st.info("No valid date data found. Showing all data.")
        else:
            filtered_df = df
            st.info("Date column not found. Showing all data.")
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Sales Distribution", "Time Series Analysis", "Correlation Analysis"])
        
        with tab1:
            st.subheader("Sales Distribution")
            if 'EV_Sales_Quantity' in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(filtered_df['EV_Sales_Quantity'], kde=True, ax=ax)
                ax.set_title('Distribution of EV Sales')
                ax.set_xlabel('Sales')
                st.pyplot(fig)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{filtered_df['EV_Sales_Quantity'].mean():.2f}")
                col2.metric("Median", f"{filtered_df['EV_Sales_Quantity'].median():.2f}")
                col3.metric("Std Dev", f"{filtered_df['EV_Sales_Quantity'].std():.2f}")
                col4.metric("Range", f"{filtered_df['EV_Sales_Quantity'].max() - filtered_df['EV_Sales_Quantity'].min():.2f}")
            else:
                st.info("Sales column not found.")
        
        with tab2:
            st.subheader("Time Series Analysis")
            if 'Date' in filtered_df.columns and 'EV_Sales_Quantity' in filtered_df.columns:
                # Time aggregation
                time_agg = st.selectbox(
                    "Select time aggregation:",
                    ["Day", "Week", "Month", "Quarter", "Year"],
                    index=2  # Default to Month
                )
                
                agg_map = {
                    "Day": 'D',
                    "Week": 'W',
                    "Month": 'M',
                    "Quarter": 'Q',
                    "Year": 'Y'
                }
                
                # Aggregate data
                time_series = filtered_df.groupby(pd.Grouper(key='Date', freq=agg_map[time_agg]))['EV_Sales_Quantity'].sum().reset_index()
                
                # Plot
                fig = px.line(
                    time_series,
                    x='Date',
                    y='EV_Sales_Quantity',
                    title=f'EV Sales Trend ({time_agg}ly)',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Moving average
                if len(time_series) > 3:
                    show_ma = st.checkbox("Show moving average")
                    if show_ma:
                        ma_window = st.slider("Moving average window:", 2, min(10, len(time_series)), 3)
                        time_series['MA'] = time_series['EV_Sales_Quantity'].rolling(window=ma_window).mean()
                        
                        fig = px.line(
                            time_series,
                            x='Date',
                            y=['EV_Sales_Quantity', 'MA'],
                            title=f'EV Sales Trend with {ma_window}-period Moving Average',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Date or Sales column not found.")
        
        with tab3:
            st.subheader("Correlation Analysis")
            # Select only numeric columns
            numeric_df = filtered_df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                # Calculate correlation matrix
                corr_matrix = numeric_df.corr()
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Matrix')
                st.pyplot(fig)
                
                # Pairplot for selected variables
                st.subheader("Pairwise Relationships")
                # Convert pandas Index to list for default values
                numeric_cols_list = list(numeric_df.columns)
                default_vars = numeric_cols_list[:min(3, len(numeric_cols_list))]
                selected_vars = st.multiselect(
                    "Select variables for pairplot (max 5 recommended):",
                    numeric_cols_list,
                    default=default_vars
                )
                
                if selected_vars and len(selected_vars) >= 2:
                    if len(selected_vars) <= 5:
                        fig = sns.pairplot(filtered_df[selected_vars])
                        st.pyplot(fig)
                    else:
                        st.warning("Too many variables selected. Please select 5 or fewer for better visualization.")
            else:
                st.info("Not enough numeric columns for correlation analysis.")
    else:
        st.info("Please upload data to view sales analysis.")


def display_state_comparison(df):
    """Display state comparison page with interactive state selection."""
    st.title("State Comparison")
    
    # Get state columns (one-hot encoded)
    state_cols = [col for col in df.columns if col.startswith('State_')]
    
    if df is not None and state_cols:
        # State selection
        all_states = [col.replace('State_', '') for col in state_cols]
        default_states = all_states[:min(5, len(all_states))]
        selected_states = st.multiselect(
            "Select states to compare:",
            all_states,
            default=default_states
        )
        
        if selected_states:
            # Filter data for selected states
            state_df = df[df[[f'State_{state}' for state in selected_states]].any(axis=1)]
            
            # Comparison tabs
            tab1, tab2 = st.tabs(["Sales Comparison", "Trend Comparison"])
            
            with tab1:
                st.subheader("Sales Comparison")
                if 'EV_Sales_Quantity' in df.columns:
                    # Aggregate sales by state
                    state_sales = []
                    for state in selected_states:
                        state_data = df[df[f'State_{state}'] == 1]
                        total_sales = state_data['EV_Sales_Quantity'].sum()
                        state_sales.append({'State': state, 'EV_Sales_Quantity': total_sales})
                    
                    state_sales_df = pd.DataFrame(state_sales)
                    
                    # Bar chart
                    fig = px.bar(
                        state_sales_df,
                        x='State',
                        y='EV_Sales_Quantity',
                        title='Total EV Sales by State',
                        color='State'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Pie chart
                    fig = px.pie(
                        state_sales_df,
                        values='EV_Sales_Quantity',
                        names='State',
                        title='Sales Distribution by State'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sales column not found.")
            
            with tab2:
                st.subheader("Trend Comparison")
                if 'Date' in df.columns and 'EV_Sales_Quantity' in df.columns:
                    # Time aggregation
                    time_agg = st.selectbox(
                        "Select time aggregation:",
                        ["Month", "Quarter", "Year"],
                        index=0  # Default to Month
                    )
                    
                    agg_map = {
                        "Month": 'M',
                        "Quarter": 'Q',
                        "Year": 'Y'
                    }
                    
                    # Convert to datetime
                    state_df_copy = state_df.copy()
                    state_df_copy['Date'] = pd.to_datetime(state_df_copy['Date'], errors='coerce')
                    # Filter out any invalid dates
                    valid_state_data = state_df_copy.dropna(subset=['Date', 'EV_Sales_Quantity'])
                    
                    if not valid_state_data.empty:
                        # Create state labels for the data
                        valid_state_data['State'] = 'Unknown'
                        for state in selected_states:
                            state_mask = valid_state_data[f'State_{state}'] == 1
                            valid_state_data.loc[state_mask, 'State'] = state
                        
                        # Aggregate data by state and time
                        state_time_series = valid_state_data.groupby(['State', pd.Grouper(key='Date', freq=agg_map[time_agg])])['EV_Sales_Quantity'].sum().reset_index()
                        
                        # Line chart
                        fig = px.line(
                            state_time_series,
                            x='Date',
                            y='EV_Sales_Quantity',
                            color='State',
                            title=f'EV Sales Trend by State ({time_agg}ly)',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid date data available for trend comparison")
                else:
                    st.info("Date or Sales column not found.")
        else:
            st.info("Please select at least one state to compare.")
    else:
        st.info("State data not found in the dataset.")


def display_predictions(df, model_path, preprocessor_path):
    """Display predictions page with model results and prediction interface."""
    st.title("Sales Predictions")
    # Define project_root for path resolution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Load pipeline
    pipeline_path = os.path.join(project_root, "models", "best_pipeline.pkl")
    pipeline = None
    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
    else:
        st.error(f"Pipeline file not found at {pipeline_path}. Please train the model pipeline first.")
        st.stop()

    if df is not None:
        if pipeline is not None:
            st.subheader("Model Information")
            st.write(f"Pipeline Type: {type(pipeline).__name__}")

            # Prediction interface
            st.subheader("Make Predictions")
            st.write("Enter values for the features to predict EV sales:")

            # Exclude target and non-feature columns
            exclude_cols = ['EV_Sales_Quantity', 'Date']
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # Group features by type for better organization
            state_cols = [col for col in feature_cols if col.startswith('State_')]
            vehicle_cols = [col for col in feature_cols if col.startswith('Vehicle_')]
            month_cols = [col for col in feature_cols if col.startswith('Month_Name_')]
            other_cols = [col for col in feature_cols if col not in state_cols + vehicle_cols + month_cols]

            # --- Collect raw feature values for prediction ---
            raw_input_data = {}

            # State selection (raw)
            if state_cols:
                st.subheader("State Selection")
                selected_state = st.selectbox("Select State:", [col.replace('State_', '') for col in state_cols])
                raw_input_data['State'] = selected_state

            # Vehicle Category selection (raw)
            vehicle_category_cols = [col for col in vehicle_cols if col.startswith('Vehicle_Category_')]
            if vehicle_category_cols:
                st.subheader("Vehicle Category Selection")
                selected_vehicle_category = st.selectbox("Select Vehicle Category:", [col.replace('Vehicle_Category_', '') for col in vehicle_category_cols])
                raw_input_data['Vehicle_Category'] = selected_vehicle_category

            # Vehicle Type selection (raw)
            vehicle_type_cols = [col for col in vehicle_cols if col.startswith('Vehicle_Type_')]
            if vehicle_type_cols:
                st.subheader("Vehicle Type Selection")
                selected_vehicle_type = st.selectbox("Select Vehicle Type:", [col.replace('Vehicle_Type_', '') for col in vehicle_type_cols])
                raw_input_data['Vehicle_Type'] = selected_vehicle_type

            # Vehicle Class selection (raw)
            vehicle_class_cols = [col for col in vehicle_cols if col.startswith('Vehicle_Class_')]
            if vehicle_class_cols:
                st.subheader("Vehicle Class Selection")
                selected_vehicle_class = st.selectbox("Select Vehicle Class:", [col.replace('Vehicle_Class_', '') for col in vehicle_class_cols])
                raw_input_data['Vehicle_Class'] = selected_vehicle_class

            # Month selection (raw)
            if month_cols:
                st.subheader("Month Selection")
                selected_month = st.selectbox("Select Month:", [col.replace('Month_Name_', '') for col in month_cols])
                raw_input_data['Month_Name'] = selected_month

            # Other numeric features (raw)
            if other_cols:
                st.subheader("Other Features")
                for col in other_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        raw_input_data[col] = st.slider(f"{col}:", min_val, max_val, (min_val + max_val) / 2)
                    else:
                        raw_input_data[col] = st.selectbox(f"Select {col}:", df[col].unique())

            # Make prediction
            if st.button("Predict Sales"):
                try:
                    input_df = pd.DataFrame([raw_input_data])
                    prediction = pipeline.predict(input_df)[0]
                    st.success(f"Predicted EV Sales Quantity: {prediction:,.2f}")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

        else:
            st.error("Pipeline not found. Please ensure the pipeline file exists.")
            st.info(f"Pipeline path: {pipeline_path}")
            st.info("Please run the training pipeline first using `python run.py --train`")
    else:
        st.info("Please upload data to make predictions.")


if __name__ == "__main__":
    main()