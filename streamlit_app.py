import sys
import traceback

try:
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
    
    # ===== EXACT FUNCTIONS FROM YOUR main.py =====
    
    def detect_time_column(df):
        """
        Detect potential time/date columns in the DataFrame.
        Returns column name if found, None otherwise.
        """
        time_keywords = ['date', 'time', 'month', 'year', 'day', 'quarter', 
                         'week', 'period', 'timestamp', 'datetime']
        
        for column in df.columns:
            col_lower = column.lower()
            
            if any(keyword in col_lower for keyword in time_keywords):
                return column
            
            # Check if column is datetime type
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                return column
                
            # Infer from data content
            try:
                sample_value = str(df[column].dropna().iloc[0])
                
                # Common date patterns
                import re
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                    r'[A-Za-z]{3,}',        # Month names 
                    r'Q\d',                 # Quarters (Q1, Q2, etc.)
                ]
                
                for pattern in date_patterns:
                    if re.match(pattern, sample_value):
                        return column
                        
            except (IndexError, AttributeError):
                continue
                
        return None
    
    def create_time_series_plot(df, time_column, value_column):
        """
        Create a time-series line plot for a numeric column.
        EXACT COPY FROM main.py
        """
        try:
            # Cleaning the df
            plot_df = df[[time_column, value_column]].copy()
            plot_df = plot_df.dropna()
            
            if len(plot_df) < 2:
                return None  # Need at least 2 points for time series
            
            # Try convert time column to datetime, especially for month names
            try:
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                
                # Check the first non-null value in the time column
                first_val = str(plot_df[time_column].iloc[0]).strip().upper()
                if first_val in month_map:
                    # Convert month names to numbers and then to a date
                    plot_df['temp_month_num'] = plot_df[time_column].str.strip().str.upper().map(month_map)
                    current_year = pd.Timestamp.now().year
                    plot_df['temp_date'] = pd.to_datetime(
                        plot_df['temp_month_num'].astype(str) + '-' + str(current_year), 
                        format='%m-%Y', 
                        errors='coerce'
                    )
                    if not plot_df['temp_date'].isnull().all():
                        time_column_for_plot = 'temp_date'
                        plot_df = plot_df.sort_values('temp_date')
                    else:
                        time_column_for_plot = time_column
                else:
                    # Try to parse as datetime normally
                    plot_df[time_column] = pd.to_datetime(plot_df[time_column], errors='coerce')
                    time_column_for_plot = time_column
                    if not plot_df[time_column].isnull().all():
                        plot_df = plot_df.sort_values(time_column)
            except Exception as e:
                print(f"Error parsing time column: {e}")
                time_column_for_plot = time_column
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 5))
            
            ax.plot(plot_df[time_column_for_plot], plot_df[value_column], 
                    marker='o', linewidth=2, markersize=8)
            
            ax.set_title(f'{value_column} over {time_column}')
            ax.set_xlabel(time_column)
            ax.set_ylabel(value_column)
            
            # Formatting x-axis for dates
            if time_column_for_plot == 'temp_date' or pd.api.types.is_datetime64_any_dtype(plot_df[time_column_for_plot]):
                ax.tick_params(axis='x', rotation=45)
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
            else:
                ax.tick_params(axis='x', rotation=45)
            
            ax.grid(True, alpha=0.3)
            
            # Add trend line if enough points
            if len(plot_df) >= 3:
                # Calculate linear trend
                x_numeric = np.arange(len(plot_df))
                y_values = plot_df[value_column].values
                
                # Fit linear regression to show regular trend
                z = np.polyfit(x_numeric, y_values, 1)
                p = np.poly1d(z)
                
                # Plot trend line
                ax.plot(plot_df[time_column_for_plot], p(x_numeric), 
                       'r--', alpha=0.7, label='Trend')
                ax.legend()
            
            return fig
                
        except Exception as e:
            st.error(f"Error creating time series plot: {e}")
            return None
    
    def create_missing_chart(df):
        # Create bar chart of missing values per column
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(missing_pct.index, missing_pct.values, color='coral', edgecolor='black')
        ax.set_title('Missing Values Percentage by Column')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Missing Percentage (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        return fig
    
    def create_correlation_heatmap(df):
        """Create correlation heatmap for numerical columns"""
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) < 2:
            return None  # Need at least 2 numerical columns
        
        corr_matrix = numerical_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap')
        return fig
    
    def create_histogram(data, column):
        # Create histogram for numerical column
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data.dropna(), bins='auto', edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        return fig
    
    def create_boxplot(data, column):
        # Create box plot for skewed data or with outliers
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data.dropna())
        ax.set_title(f'Box Plot of {column}')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        return fig
    
    def create_bar_chart(data, column):
        # Create bar chart for categorical data
        value_counts = data.value_counts().head(20)  # Limit to top 20 categories
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Top Categories in {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        return fig
    
    def generate_one_line_insight(col_data, col_name, col_type):
        """Generate one-line insight based on statistics - FROM main.py"""
        if col_type == 'numerical':
            skewness = col_data.get('enhanced_metrics', {}).get('skewness', 0)
            cv = col_data.get('enhanced_metrics', {}).get('coefficient_of_variation', 0)
            missing_pct = col_data.get('enhanced_metrics', {}).get('missing_percentage', 0)
            
            insights = []
            if abs(skewness) > 1:
                insights.append("Highly skewed distribution")
            elif abs(skewness) > 0.5:
                insights.append("Moderately skewed")
                
            if cv > 0.5:
                insights.append("High relative variability")
                
            if missing_pct > 0.2:
                insights.append(f"Significant missing data ({missing_pct:.1%})")
            
            return f"{col_name}: " + ("; ".join(insights) if insights else "Normal distribution, good data quality")
        
        elif col_type == 'categorical':
            unique_count = col_data.get('unique_count', 0)
            if unique_count > 50:
                return f"{col_name}: High cardinality ({unique_count} unique values)"
            else:
                return f"{col_name}: {unique_count} unique categories"
    
    # ===== MAIN APP LOGIC =====
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
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
            st.info(f"⏰ Time dimension detected: '{time_column}'. Consider analyzing trends over time.")
        
        # Analysis button
        if st.button("🚀 Run Automated EDA", type="primary"):
            with st.spinner("Analyzing data and generating visualizations..."):
                
                # Initialize results
                analysis_results = {}
                insights = []
                
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
                
                # Analyze each column (EXACT LOGIC FROM main.py)
                for column in df.columns:
                    col_type = 'categorical' if df[column].dtype == 'object' else 'numerical'
                    col_data_series = df[column]
                    
                    if col_type == 'numerical':
                        col_data = col_data_series.dropna()
                        
                        if len(col_data) > 0:
                            # Calculate EXACT statistics from main.py
                            stats_dict = {
                                "basic_statistics": {
                                    "count": int(col_data.count()),
                                    "mean": float(col_data.mean()),
                                    "std": float(col_data.std()),
                                    "min": float(col_data.min()),
                                    "25%": float(np.percentile(col_data, 25)),
                                    "50%": float(col_data.median()),
                                    "75%": float(np.percentile(col_data, 75)),
                                    "max": float(col_data.max())
                                },
                                "enhanced_metrics": {
                                    "skewness": float(col_data.skew()),
                                    "coefficient_of_variation": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else float('inf'),
                                    "missing_values": int(df[column].isnull().sum()),
                                    "missing_percentage": float(df[column].isnull().mean()),
                                    "outlier_count": int(((col_data_series - col_data.mean()).abs() > 3 * col_data.std()).sum())
                                }
                            }
                            
                            analysis_results[column] = stats_dict
                            
                            # Generate visualizations
                            if not time_column or column == time_column:
                                hist_fig = create_histogram(col_data_series, column)
                                st.pyplot(hist_fig)
                                
                                skewness = abs(stats_dict['enhanced_metrics']['skewness'])
                                outliers = stats_dict['enhanced_metrics']['outlier_count']
                                
                                if skewness > 0.5 or outliers > 0:
                                    box_fig = create_boxplot(col_data_series, column)
                                    st.pyplot(box_fig)
                            
                            # Time-series plot if time column detected and this isn't the time column
                            if time_column and column != time_column:
                                ts_fig = create_time_series_plot(df, time_column, column)
                                if ts_fig:
                                    st.subheader(f"⏰ {column} over {time_column}")
                                    st.pyplot(ts_fig)
                            
                            # Generate insight
                            insights.append(generate_one_line_insight(stats_dict, column, 'numerical'))
                    
                    else:  # categorical
                        unique_count = col_data_series.nunique()
                        stats_dict = {
                            "unique_count": int(unique_count),
                            "missing_values": int(df[column].isnull().sum()),
                            "missing_percentage": float(df[column].isnull().mean()),
                            "top_categories": col_data_series.value_counts().head(5).to_dict()
                        }
                        
                        analysis_results[column] = stats_dict
                        
                        # Categorical time-series if it's the time column
                        if time_column == column:
                            bar_fig = create_bar_chart(col_data_series, column)
                            st.pyplot(bar_fig)
                        
                        # Generate visualization for categorical (if not too many categories)
                        if unique_count <= 50 and unique_count > 1:
                            bar_fig = create_bar_chart(col_data_series, column)
                            st.pyplot(bar_fig)
                        
                        # Generate insight
                        insights.append(generate_one_line_insight(stats_dict, column, 'categorical'))
                
                # Show detailed statistics
                st.subheader("📊 Detailed Statistics")
                
                for col_name, col_data in analysis.items():
                    with st.expander(f"**{col_name}**"):
                        if 'basic_statistics' in col_data:
                            # Numerical column - show ALL stats from main.py
                            st.write("**Basic Statistics:**")
                            stats = col_data['basic_statistics']
                            cols = st.columns(4)
                            with cols[0]:
                                st.metric("Count", f"{stats['count']:,}")
                                st.metric("Mean", f"{stats['mean']:.4f}")
                            with cols[1]:
                                st.metric("Std Dev", f"{stats['std']:.4f}")
                                st.metric("Min", f"{stats['min']:.4f}")
                            with cols[2]:
                                st.metric("25%", f"{stats['25%']:.4f}")
                                st.metric("50%", f"{stats['50%']:.4f}")
                            with cols[3]:
                                st.metric("75%", f"{stats['75%']:.4f}")
                                st.metric("Max", f"{stats['max']:.4f}")
                            
                            # Enhanced metrics
                            st.write("**Enhanced Metrics:**")
                            enhanced = col_data['enhanced_metrics']
                            ecols = st.columns(4)
                            with ecols[0]:
                                skew_val = enhanced['skewness']
                                skew_color = "🔴" if abs(skew_val) > 1 else "🟡" if abs(skew_val) > 0.5 else "🟢"
                                st.metric("Skewness", f"{skew_val:.4f} {skew_color}")
                            with ecols[1]:
                                cv_val = enhanced['coefficient_of_variation']
                                cv_color = "🔴" if cv_val > 0.5 else "🟢"
                                st.metric("Coefficient of Variation", f"{cv_val:.4f} {cv_color}")
                            with ecols[2]:
                                st.metric("Missing Values", f"{enhanced['missing_values']:,}")
                            with ecols[3]:
                                st.metric("Missing %", f"{enhanced['missing_percentage']:.2%}")
                        else:
                            # Categorical column
                            st.write("**Categorical Statistics:**")
                            cat_cols = st.columns(3)
                            with cat_cols[0]:
                                st.metric("Unique Values", col_data['unique_count'])
                            with cat_cols[1]:
                                st.metric("Missing Values", f"{col_data['missing_values']:,}")
                            with cat_cols[2]:
                                st.metric("Missing %", f"{col_data['missing_percentage']:.2%}")
                            
                            # Top categories
                            st.write("**Top 5 Categories:**")
                            for cat, count in col_data['top_categories'].items():
                                st.write(f"- {cat}: {count:,}")
                
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
                    summary += f"\n{col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, Min={df[col].min():.2f}, Max={df[col].max():.2f}"
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

except Exception as e:
    # Show error page
    import streamlit as st
    st.set_page_config(page_title="Error", layout="wide")
    st.title("🚨 DataIntel - Critical Error")
    
    st.error(f"**Error Type:** {type(e).__name__}")
    st.error(f"**Error Message:** {str(e)}")
    
    st.write("### Full Traceback:")
    st.code(traceback.format_exc(), language='python')
    
    # Show what packages are installed
    st.write("### Installed Packages:")
    try:
        import pkg_resources
        packages = []
        for pkg in pkg_resources.working_set:
            packages.append(f"{pkg.key}=={pkg.version}")
        
        # Show most important ones
        important = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn']
        for pkg in important:
            matches = [p for p in packages if p.startswith(pkg)]
            if matches:
                st.write(f"- {matches[0]}")
    except:
        st.write("Could not load package list")
    
    st.stop()
