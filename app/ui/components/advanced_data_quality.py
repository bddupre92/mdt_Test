"""
Advanced Data Quality Visualization Component for the Interactive Data Configuration Dashboard.
This module provides comprehensive data quality analysis and visualization capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
import time
from datetime import datetime

# Set the style for all visualizations
plt.style.use('ggplot')

def render_advanced_data_quality(data=None):
    """
    Main function to render the advanced data quality visualization component.
    
    Parameters:
    -----------
    data : pandas.DataFrame, optional
        The dataset to analyze. If None, will try to load from session state.
    """
    st.subheader("Advanced Data Quality Analysis")
    
    # Check if data is available
    if data is None:
        if 'data' in st.session_state and st.session_state.data is not None:
            data = st.session_state.data
        else:
            st.info("Please upload data in the Pipeline Builder tab first.")
            return
    
    # Create tabs for different analyses
    quality_tabs = st.tabs([
        "Data Profile", 
        "Quality Scoring", 
        "Distribution Analysis",
        "Correlation Analysis",
        "Time Series Analysis",
        "Anomaly Detection"
    ])
    
    # Data Profile Tab
    with quality_tabs[0]:
        render_data_profile(data)
    
    # Quality Scoring Tab
    with quality_tabs[1]:
        render_quality_scoring(data)
    
    # Distribution Analysis Tab
    with quality_tabs[2]:
        render_distribution_analysis(data)
    
    # Correlation Analysis Tab
    with quality_tabs[3]:
        render_correlation_analysis(data)
    
    # Time Series Analysis Tab
    with quality_tabs[4]:
        render_time_series_analysis(data)
    
    # Anomaly Detection Tab
    with quality_tabs[5]:
        render_anomaly_detection(data)

def render_data_profile(data):
    """
    Render comprehensive data profiling information.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze.
    """
    st.write("### Data Profile")
    
    # Basic dataset information
    st.write("#### Basic Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{data.shape[1]:,}")
    with col3:
        memory_usage = data.memory_usage(deep=True).sum()
        st.metric("Memory Usage", f"{memory_usage / 1024**2:.2f} MB")
    
    # Data types detection
    st.write("#### Data Types")
    dtypes = data.dtypes.value_counts().reset_index()
    dtypes.columns = ['Data Type', 'Count']
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(dtypes)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(dtypes['Count'], labels=dtypes['Data Type'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    # Detailed column information
    st.write("#### Column Details")
    
    # Create a dataframe with column details
    column_details = []
    for col in data.columns:
        col_type = str(data[col].dtype)
        missing = data[col].isna().sum()
        missing_pct = missing / len(data) * 100
        unique = data[col].nunique()
        unique_pct = unique / len(data) * 100
        
        if pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            mean_val = data[col].mean()
            std_val = data[col].std()
            skew_val = data[col].skew()
            column_details.append({
                'Column': col,
                'Type': col_type,
                'Missing': missing,
                'Missing %': f"{missing_pct:.2f}%",
                'Unique': unique,
                'Unique %': f"{unique_pct:.2f}%",
                'Min': min_val,
                'Max': max_val,
                'Mean': mean_val,
                'Std Dev': std_val,
                'Skewness': skew_val
            })
        else:
            column_details.append({
                'Column': col,
                'Type': col_type,
                'Missing': missing,
                'Missing %': f"{missing_pct:.2f}%",
                'Unique': unique,
                'Unique %': f"{unique_pct:.2f}%",
                'Min': 'N/A',
                'Max': 'N/A',
                'Mean': 'N/A',
                'Std Dev': 'N/A',
                'Skewness': 'N/A'
            })
    
    column_df = pd.DataFrame(column_details)
    st.dataframe(column_df, height=400)
    
    # Sample data
    with st.expander("View Sample Data"):
        st.dataframe(data.head(10))

def render_quality_scoring(data):
    """
    Calculate and render data quality scores.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze.
    """
    st.write("### Data Quality Scoring")
    
    # Calculate completeness score (based on missing values)
    completeness_scores = {}
    for col in data.columns:
        missing = data[col].isna().sum()
        completeness = 1 - (missing / len(data))
        completeness_scores[col] = completeness
    
    avg_completeness = sum(completeness_scores.values()) / len(completeness_scores)
    
    # Calculate validity score (based on data types and range checks)
    validity_scores = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            # Check for outliers using IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            validity = 1 - (outliers / len(data))
        elif pd.api.types.is_string_dtype(data[col]):
            # For string columns, check for empty strings
            empty = (data[col] == '').sum()
            validity = 1 - (empty / len(data))
        else:
            validity = 1.0
        validity_scores[col] = validity
    
    avg_validity = sum(validity_scores.values()) / len(validity_scores)
    
    # Calculate consistency score (based on unique values)
    consistency_scores = {}
    for col in data.columns:
        unique_pct = data[col].nunique() / len(data)
        # For categorical columns, high uniqueness is bad
        # For numerical columns, we're more lenient
        if pd.api.types.is_numeric_dtype(data[col]):
            consistency = 1 - (unique_pct if unique_pct > 0.9 else 0)
        else:
            consistency = 1 - unique_pct
        consistency_scores[col] = max(0, consistency)
    
    avg_consistency = sum(consistency_scores.values()) / len(consistency_scores)
    
    # Calculate overall quality score
    overall_score = (avg_completeness * 0.4) + (avg_validity * 0.4) + (avg_consistency * 0.2)
    overall_score = overall_score * 100  # Convert to percentage
    
    # Display overall score
    st.write("#### Overall Quality Score")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{overall_score:.1f}%")
    with col2:
        st.metric("Completeness", f"{avg_completeness*100:.1f}%")
    with col3:
        st.metric("Validity", f"{avg_validity*100:.1f}%")
    with col4:
        st.metric("Consistency", f"{avg_consistency*100:.1f}%")
    
    # Display per-column scores
    st.write("#### Quality Scores by Column")
    
    quality_df = pd.DataFrame({
        'Column': list(completeness_scores.keys()),
        'Completeness': [completeness_scores[col] * 100 for col in completeness_scores],
        'Validity': [validity_scores[col] * 100 for col in validity_scores],
        'Consistency': [consistency_scores[col] * 100 for col in consistency_scores]
    })
    
    # Calculate overall column score
    quality_df['Overall'] = (quality_df['Completeness'] * 0.4 + 
                            quality_df['Validity'] * 0.4 + 
                            quality_df['Consistency'] * 0.2)
    
    # Sort by overall score
    quality_df = quality_df.sort_values('Overall', ascending=False)
    
    # Display as table
    st.dataframe(quality_df)
    
    # Visualize column scores
    fig, ax = plt.subplots(figsize=(10, 6))
    quality_df = quality_df.sort_values('Overall')
    
    # Plot stacked bar chart
    quality_df.plot(x='Column', y=['Completeness', 'Validity', 'Consistency'], 
                   kind='bar', stacked=False, ax=ax)
    
    ax.set_title('Quality Scores by Column')
    ax.set_xlabel('Column')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Recommendations based on scores
    st.write("#### Recommendations")
    
    recommendations = []
    
    # Find columns with low completeness
    low_completeness = quality_df[quality_df['Completeness'] < 90]['Column'].tolist()
    if low_completeness:
        recommendations.append(f"Consider handling missing values in columns: {', '.join(low_completeness)}")
    
    # Find columns with low validity
    low_validity = quality_df[quality_df['Validity'] < 90]['Column'].tolist()
    if low_validity:
        recommendations.append(f"Check for outliers or invalid values in columns: {', '.join(low_validity)}")
    
    # Find columns with low consistency
    low_consistency = quality_df[quality_df['Consistency'] < 70]['Column'].tolist()
    if low_consistency:
        recommendations.append(f"Review high cardinality in columns: {', '.join(low_consistency)}")
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")
    else:
        st.write("No specific recommendations. Your data quality looks good!")

def render_distribution_analysis(data):
    """
    Analyze and visualize the distributions of numerical features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze.
    """
    st.write("### Distribution Analysis")
    
    # Get numerical columns
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    if not numerical_cols:
        st.info("No numerical columns found in the dataset.")
        return
    
    # Select column for analysis
    selected_col = st.selectbox("Select column for distribution analysis", numerical_cols)
    
    # Distribution plot
    st.write(f"#### Distribution of {selected_col}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data[selected_col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Histogram of {selected_col}')
        st.pyplot(fig)
    
    with col2:
        # Box plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=data[selected_col].dropna(), ax=ax)
        ax.set_title(f'Box Plot of {selected_col}')
        st.pyplot(fig)
    
    # Statistical tests
    st.write("#### Statistical Tests")
    
    # Descriptive statistics
    desc_stats = data[selected_col].describe()
    st.write("Descriptive Statistics:")
    st.write(desc_stats)
    
    # Normality test
    st.write("Normality Test (Shapiro-Wilk):")
    
    # Only use a sample for Shapiro-Wilk test if the dataset is large
    sample_data = data[selected_col].dropna()
    if len(sample_data) > 5000:
        sample_data = sample_data.sample(5000, random_state=42)
    
    try:
        stat, p_value = stats.shapiro(sample_data)
        st.write(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.write("Result: The data does not follow a normal distribution (p < 0.05)")
        else:
            st.write("Result: The data follows a normal distribution (p >= 0.05)")
    except Exception as e:
        st.write(f"Could not perform Shapiro-Wilk test: {str(e)}")
    
    # QQ Plot
    st.write("#### Q-Q Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(data[selected_col].dropna(), plot=ax)
    ax.set_title(f'Q-Q Plot of {selected_col}')
    st.pyplot(fig)

def render_correlation_analysis(data):
    """
    Analyze and visualize correlations between features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze.
    """
    st.write("### Correlation Analysis")
    
    # Get numerical columns
    numerical_data = data.select_dtypes(include=['number'])
    
    if numerical_data.shape[1] < 2:
        st.info("Need at least two numerical columns for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()
    
    # Visualization options
    viz_type = st.radio("Visualization Type", ["Heatmap", "Pairplot", "Feature Correlation"])
    
    if viz_type == "Heatmap":
        st.write("#### Correlation Heatmap")
        
        # Mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
        
        plt.title('Feature Correlation Heatmap')
        st.pyplot(fig)
        
    elif viz_type == "Pairplot":
        st.write("#### Pairplot")
        
        # Let user select columns for pairplot (limit to prevent performance issues)
        if numerical_data.shape[1] > 5:
            selected_cols = st.multiselect(
                "Select columns for pairplot (max 5 recommended)", 
                numerical_data.columns.tolist(),
                default=numerical_data.columns.tolist()[:4]
            )
            
            if not selected_cols:
                st.info("Please select at least two columns.")
                return
            
            if len(selected_cols) > 5:
                st.warning("Too many columns selected. This may cause performance issues.")
            
            plot_data = data[selected_cols]
        else:
            plot_data = numerical_data
        
        # Create pairplot
        with st.spinner("Generating pairplot..."):
            fig = sns.pairplot(plot_data, diag_kind='kde')
            plt.suptitle('Pairplot of Selected Features', y=1.02)
            st.pyplot(fig)
    
    elif viz_type == "Feature Correlation":
        st.write("#### Feature Correlation")
        
        # Let user select target column
        target_col = st.selectbox("Select target column", numerical_data.columns.tolist())
        
        # Calculate correlations with target
        correlations = numerical_data.corr()[target_col].sort_values(ascending=False)
        correlations = correlations.drop(target_col)  # Remove self-correlation
        
        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(correlations) * 0.3)))
        correlations.plot(kind='barh', ax=ax)
        ax.set_title(f'Correlation with {target_col}')
        ax.set_xlabel('Correlation Coefficient')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display correlation values
        st.write("Correlation Values:")
        corr_df = pd.DataFrame({'Correlation': correlations})
        st.dataframe(corr_df)

def render_time_series_analysis(data):
    """
    Analyze time series data if available.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze.
    """
    st.write("### Time Series Analysis")
    
    # Check if there are datetime columns
    datetime_cols = []
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_string_dtype(data[col]):
            # Try to convert string columns to datetime
            try:
                pd.to_datetime(data[col])
                datetime_cols.append(col)
            except:
                pass
    
    if not datetime_cols:
        st.info("No datetime columns detected in the dataset. Please convert date columns to datetime format for time series analysis.")
        return
    
    # Select datetime column
    date_col = st.selectbox("Select date/time column", datetime_cols)
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        try:
            data = data.copy()
            data[date_col] = pd.to_datetime(data[date_col])
        except Exception as e:
            st.error(f"Error converting to datetime: {str(e)}")
            return
    
    # Select value column
    value_cols = data.select_dtypes(include=['number']).columns.tolist()
    if not value_cols:
        st.info("No numerical columns found for time series analysis.")
        return
    
    value_col = st.selectbox("Select value column", value_cols)
    
    # Sort data by date
    data_sorted = data.sort_values(by=date_col)
    
    # Time series plot
    st.write(f"#### Time Series Plot: {value_col} over {date_col}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_sorted[date_col], data_sorted[value_col])
    ax.set_title(f'{value_col} over Time')
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Resampling options
    st.write("#### Resampled View")
    
    resample_options = {
        'Day': 'D',
        'Week': 'W',
        'Month': 'M',
        'Quarter': 'Q',
        'Year': 'Y'
    }
    
    resample_period = st.selectbox("Resample by", list(resample_options.keys()))
    agg_method = st.selectbox("Aggregation method", ['Mean', 'Sum', 'Min', 'Max', 'Count'])
    
    # Map to pandas methods
    agg_map = {
        'Mean': 'mean',
        'Sum': 'sum',
        'Min': 'min',
        'Max': 'max',
        'Count': 'count'
    }
    
    # Resample data
    try:
        # Set date as index for resampling
        data_indexed = data_sorted.set_index(date_col)
        resampled = data_indexed[value_col].resample(resample_options[resample_period]).agg(agg_map[agg_method])
        
        # Plot resampled data
        fig, ax = plt.subplots(figsize=(12, 6))
        resampled.plot(ax=ax)
        ax.set_title(f'{value_col} ({agg_method}) by {resample_period}')
        ax.set_xlabel(date_col)
        ax.set_ylabel(f'{agg_method} of {value_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show resampled data
        st.write(f"Resampled Data ({agg_method} by {resample_period}):")
        st.dataframe(resampled.reset_index())
        
    except Exception as e:
        st.error(f"Error during resampling: {str(e)}")

def render_anomaly_detection(data):
    """
    Detect and visualize anomalies in the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze.
    """
    st.write("### Anomaly Detection")
    
    # Get numerical columns
    numerical_data = data.select_dtypes(include=['number'])
    
    if numerical_data.shape[1] < 1:
        st.info("No numerical columns found for anomaly detection.")
        return
    
    # Select columns for anomaly detection
    selected_cols = st.multiselect(
        "Select columns for anomaly detection", 
        numerical_data.columns.tolist(),
        default=numerical_data.columns.tolist()[:min(5, len(numerical_data.columns))]
    )
    
    if not selected_cols:
        st.info("Please select at least one column.")
        return
    
    # Parameters for anomaly detection
    contamination = st.slider("Contamination (expected proportion of outliers)", 0.01, 0.5, 0.1, 0.01)
    
    # Detect anomalies
    st.write("#### Anomaly Detection Results")
    
    with st.spinner("Detecting anomalies..."):
        # Prepare data
        X = numerical_data[selected_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        y_pred = model.fit_predict(X_scaled)
        
        # Convert predictions to anomaly labels (1: normal, -1: anomaly)
        anomalies = y_pred == -1
        
        # Add anomaly column to data
        anomaly_data = data.copy()
        anomaly_data['is_anomaly'] = anomalies
        
        # Show anomaly statistics
        num_anomalies = anomalies.sum()
        st.metric("Detected Anomalies", f"{num_anomalies} ({num_anomalies/len(data)*100:.2f}%)")
        
        # Display sample of anomalies
        st.write("Sample of Detected Anomalies:")
        st.dataframe(anomaly_data[anomalies].head(10))
        
        # Feature contribution to anomalies
        st.write("#### Feature Contribution to Anomalies")
        
        # Calculate mean and std for normal and anomaly points
        normal_means = X[~anomalies].mean()
        anomaly_means = X[anomalies].mean()
        normal_stds = X[~anomalies].std()
        
        # Calculate z-scores for anomalies
        z_scores = (anomaly_means - normal_means) / normal_stds
        
        # Plot feature contribution
        fig, ax = plt.subplots(figsize=(10, 6))
        z_scores.abs().sort_values().plot(kind='barh', ax=ax)
        ax.set_title('Feature Contribution to Anomalies')
        ax.set_xlabel('Absolute Z-Score')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualize anomalies in 2D
        if len(selected_cols) >= 2:
            st.write("#### Anomaly Visualization")
            
            # Select two features for visualization
            viz_cols = st.multiselect(
                "Select two features for visualization", 
                selected_cols,
                default=selected_cols[:min(2, len(selected_cols))]
            )
            
            if len(viz_cols) == 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    data[viz_cols[0]], 
                    data[viz_cols[1]], 
                    c=anomalies, 
                    cmap='coolwarm', 
                    alpha=0.6
                )
                ax.set_title('Anomaly Detection Visualization')
                ax.set_xlabel(viz_cols[0])
                ax.set_ylabel(viz_cols[1])
                plt.colorbar(scatter, label='Anomaly')
                plt.tight_layout()
                st.pyplot(fig)
            elif len(viz_cols) != 0:
                st.info("Please select exactly two features for visualization.")
