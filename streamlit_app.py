# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from pathlib import Path
import uuid
import tempfile

# Page config
st.set_page_config(
    page_title="DataIntel: EDA Automation",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 DataIntel: EDA Automation")
st.markdown("Upload your CSV file for automated exploratory data analysis with intelligent visualizations")

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'plots' not in st.session_state:
    st.session_state.plots = []
if 'insights' not in st.session_state:
    st.session_state.insights = []

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

# Function from your FastAPI app (copied and adapted)
def detect_time_column(df):
    time_keywords = ['date', 'time', 'month', 'year', 'day', 'quarter', 
                     'week', 'period', 'timestamp', 'datetime']
    
    for column in df.columns:
        col_lower = column.lower()
        
        if any(keyword in col_lower for keyword in time_keywords):
            return column
        
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
            
        try:
            sample_value = str(df[column].dropna().iloc[0])
            import re
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}',
                r'[A-Za-z]{3,}',
                r'Q\d',
            ]
            
            for pattern in date_patterns:
                if re.match(pattern, sample_value):
                    return column
                    
        except (IndexError, AttributeError):
            continue
            
    return None

def create_time_series_plot(df, time_column, value_column):
    try:
        plot_df = df[[time_column, value_column]].copy()
        plot_df = plot_df.dropna()
        
        if len(plot_df) < 2:
            return None
        
        # Try to convert to datetime
        try:
            plot_df[time_column] = pd.to_datetime(plot_df[time_column], errors='coerce')
            if not plot_df[time_column].isnull().all():
                plot_df = plot_df.sort_values(time_column)
        except:
            pass
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_df[time_column], plot_df[value_column], marker='o', linewidth=2)
        ax.set_title(f'{value_column} over {time_column}')
        ax.set_xlabel(time_column)
        ax.set_ylabel(value_column)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        return fig
    except Exception as e:
        st.error(f"Error creating time series: {e}")
        return None

def create_missing_chart(df):
    missing_pct = (df.isnull().sum() / len(df)) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(missing_pct.index, missing_pct.values, color='coral', edgecolor='black')
    ax.set_title('Missing Values Percentage by Column')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Missing Percentage (%)')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    return fig

def create_correlation_heatmap(df):
    numerical_df = df.select_dtypes(include=[np.number])
    if len(numerical_df.columns) < 2:
        return None
    
    corr_matrix = numerical_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

def analyze_data(df):
    """Main analysis function adapted from FastAPI"""
    insights = []
    time_column = detect_time_column(df)
    
    if time_column:
        insights.append(f"⏰ Time dimension detected: '{time_column}'")
    
    return insights, time_column

# Main analysis logic
if uploaded_file is not None:
    # Read the file
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ File loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Basic info
    with st.expander("📋 Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Column types
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Columns", df.shape[1])
    with col2:
        st.metric("Total Rows", df.shape[0])
    
    # Detect time column
    time_column = detect_time_column(df)
    
    if time_column:
        st.info(f"🕒 Time series detected! Using '{time_column}' as time dimension")
    
    # Analysis button
    if st.button("🚀 Run Automated EDA", type="primary"):
        with st.spinner("Analyzing data and generating visualizations..."):
            
            # 1. Missing values chart
            st.subheader("📊 Missing Values Analysis")
            missing_fig = create_missing_chart(df)
            if missing_fig:
                st.pyplot(missing_fig)
            
            # 2. Correlation heatmap
            st.subheader("🔥 Correlation Heatmap")
            corr_fig = create_correlation_heatmap(df)
            if corr_fig:
                st.pyplot(corr_fig)
            else:
                st.info("Need at least 2 numerical columns for correlation heatmap")
            
            # 3. Time series plots if time column exists
            if time_column:
                st.subheader(f"⏰ Time Series Analysis (over {time_column})")
                # Get numerical columns (excluding time column itself)
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if time_column in numerical_cols:
                    numerical_cols.remove(time_column)
                
                # Limit to 4 columns for display
                for col in numerical_cols[:4]:
                    ts_fig = create_time_series_plot(df, time_column, col)
                    if ts_fig:
                        st.pyplot(ts_fig)
            
            # 4. Column statistics
            st.subheader("📈 Column Statistics")
            stats_col1, stats_col2 = st.columns(2)
            
            all_cols = df.columns.tolist()
            mid_point = len(all_cols) // 2
            
            with stats_col1:
                for col in all_cols[:mid_point]:
                    with st.expander(f"**{col}**"):
                        if df[col].dtype in [np.int64, np.float64]:
                            st.metric("Type", "Numerical")
                            st.metric("Mean", f"{df[col].mean():.2f}")
                            st.metric("Std Dev", f"{df[col].std():.2f}")
                            st.metric("Missing", f"{df[col].isnull().sum()} ({df[col].isnull().mean():.1%})")
                        else:
                            st.metric("Type", "Categorical")
                            st.metric("Unique Values", df[col].nunique())
                            st.metric("Missing", f"{df[col].isnull().sum()} ({df[col].isnull().mean():.1%})")
            
            with stats_col2:
                for col in all_cols[mid_point:]:
                    with st.expander(f"**{col}**"):
                        if df[col].dtype in [np.int64, np.float64]:
                            st.metric("Type", "Numerical")
                            st.metric("Mean", f"{df[col].mean():.2f}")
                            st.metric("Std Dev", f"{df[col].std():.2f}")
                            st.metric("Missing", f"{df[col].isnull().sum()} ({df[col].isnull().mean():.1%})")
                        else:
                            st.metric("Type", "Categorical")
                            st.metric("Unique Values", df[col].nunique())
                            st.metric("Missing", f"{df[col].isnull().sum()} ({df[col].isnull().mean():.1%})")
            
            st.session_state.analysis_done = True
    
    # Download option
    if st.session_state.analysis_done:
        st.divider()
        st.subheader("📥 Export Analysis")
        
        # Create summary stats
        summary = f"""
        DataIntel EDA Report
        File: {uploaded_file.name}
        Shape: {df.shape[0]} rows × {df.shape[1]} columns
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Key Statistics:
        """
        
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                summary += f"\n{col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}"
            else:
                summary += f"\n{col}: {df[col].nunique()} unique values"
        
        st.download_button(
            label="📄 Download Report Summary",
            data=summary,
            file_name=f"eda_report_{uploaded_file.name.split('.')[0]}.txt",
            mime="text/plain"
        )

# Footer
st.divider()
st.caption("DataIntel: EDA Automation • [GitHub Repository](https://github.com/kous6942/DataIntel_Automated-EDA)")