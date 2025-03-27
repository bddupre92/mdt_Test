"""
Data Configuration Component

This module provides the data configuration dashboard component for the MoE framework.
It includes a visual pipeline builder, advanced visualization, and template management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set matplotlib style
plt.style.use('ggplot')

# Import preprocessing pipeline
from data.preprocessing_pipeline import (
    PreprocessingPipeline, 
    MissingValueHandler, 
    OutlierHandler, 
    FeatureScaler, 
    CategoryEncoder, 
    FeatureSelector,
    TimeSeriesProcessor
)

# Import preprocessing manager for advanced pipeline
from data.preprocessing_manager import PreprocessingManager

def render_data_configuration():
    """Render the enhanced data configuration dashboard with all components."""
    st.header("Interactive Data Configuration Dashboard")
    
    # Import the preprocessing manager and components
    from data.preprocessing_manager import PreprocessingManager
    
    # Create tabs for different sections
    tabs = st.tabs([
        "Pipeline Builder", 
        "Advanced Pipeline",
        "Advanced Data Quality", 
        "Template Management"
    ])
    
    # Pipeline Builder Tab
    with tabs[0]:
        # Check if we should use the drag-drop pipeline builder
        use_drag_drop = st.session_state.get('use_drag_drop_builder', False)
        
        if use_drag_drop:
            # Import and use the drag-drop pipeline builder
            from app.ui.components.drag_drop_pipeline import render_drag_drop_pipeline_builder
            render_drag_drop_pipeline_builder()
        else:
            # Use the standard pipeline builder with an option to switch
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader("Pipeline Builder")
            with col2:
                if st.button("Switch to Visual Builder"):
                    st.session_state.use_drag_drop_builder = True
                    st.experimental_rerun()
            with col3:
                if st.button("Switch to Advanced Pipeline"):
                    st.session_state.active_tab = "Advanced Pipeline"
                    st.experimental_rerun()
            
            render_pipeline_builder()
    
    # Advanced Pipeline Tab
    with tabs[1]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Advanced Preprocessing Pipeline")
        with col2:
            if st.button("Switch to Basic Pipeline"):
                st.session_state.active_tab = "Pipeline Builder"
                st.experimental_rerun()
        
        # Import and use the advanced preprocessing pipeline component
        from app.ui.components.preprocessing_pipeline_component import render_preprocessing_pipeline
        
        # Get data if available
        data = st.session_state.get('data', None)
        
        # If we have a basic pipeline, convert it to advanced pipeline
        if 'pipeline' in st.session_state and not 'preprocessing_manager' in st.session_state:
            try:
                st.session_state.preprocessing_manager = convert_to_advanced_pipeline(st.session_state.pipeline)
                st.success("Converted basic pipeline to advanced pipeline")
            except Exception as e:
                st.error(f"Error converting pipeline: {str(e)}")
        
        # Render the preprocessing pipeline component
        render_preprocessing_pipeline(data)
        
        # Add button to convert back to basic pipeline
        if 'preprocessing_manager' in st.session_state:
            if st.button("Convert to Basic Pipeline"):
                try:
                    st.session_state.pipeline = convert_to_basic_pipeline(st.session_state.preprocessing_manager)
                    st.success("Converted advanced pipeline to basic pipeline")
                except Exception as e:
                    st.error(f"Error converting pipeline: {str(e)}")
    
    # Advanced Data Quality Visualization Tab
    with tabs[2]:
        # Import and use the advanced data quality component
        from app.ui.components.advanced_data_quality import render_advanced_data_quality
        render_advanced_data_quality()
    
    # Template Management Tab
    with tabs[3]:
        render_template_management()
        
    # Set the active tab if specified
    if 'active_tab' in st.session_state:
        tab_index = {
            "Pipeline Builder": 0,
            "Advanced Pipeline": 1,
            "Advanced Data Quality": 2,
            "Template Management": 3
        }.get(st.session_state.active_tab, 0)
        
        # Use JavaScript to click on the appropriate tab
        js = f"""
        <script>
            function simulateClick() {{
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                const tab = tabs[{tab_index}];
                if (tab) {{
                    tab.click();
                }}
            }}
            setTimeout(simulateClick, 100);
        </script>
        """
        st.components.v1.html(js, height=0)
        
        # Clear the active tab to avoid infinite rerun
        del st.session_state.active_tab

def render_pipeline_builder():
    """Render the visual pipeline builder component."""
    st.subheader("Visual Pipeline Builder")
    
    # Initialize session state for pipeline if not exists
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = PreprocessingPipeline(name="Custom Pipeline")
    
    if 'pipeline_operations' not in st.session_state:
        st.session_state.pipeline_operations = []
    
    # File uploader for data
    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.write(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Display data sample
            st.subheader("Data Sample")
            st.dataframe(data.head())
            
            # Pipeline operations
            st.subheader("Pipeline Operations")
            
            # Add new operation
            with st.expander("Add New Operation", expanded=True):
                operation_type = st.selectbox(
                    "Select Operation Type",
                    [
                        "Missing Value Handler",
                        "Outlier Handler",
                        "Feature Scaler",
                        "Category Encoder",
                        "Feature Selector",
                        "Time Series Processor"
                    ]
                )
                
                # Configure operation based on type
                if operation_type == "Missing Value Handler":
                    configure_missing_value_handler()
                elif operation_type == "Outlier Handler":
                    configure_outlier_handler()
                elif operation_type == "Feature Scaler":
                    configure_feature_scaler()
                elif operation_type == "Category Encoder":
                    configure_category_encoder()
                elif operation_type == "Feature Selector":
                    configure_feature_selector()
                elif operation_type == "Time Series Processor":
                    configure_time_series_processor()
            
            # Display current pipeline
            if len(st.session_state.pipeline_operations) > 0:
                st.subheader("Current Pipeline")
                
                for i, op in enumerate(st.session_state.pipeline_operations):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"{i+1}. {op['type']}: {op['description']}")
                    with col2:
                        if st.button("Edit", key=f"edit_{i}"):
                            st.session_state.edit_operation_index = i
                            st.session_state.edit_operation = True
                    with col3:
                        if st.button("Remove", key=f"remove_{i}"):
                            st.session_state.pipeline_operations.pop(i)
                            if 'pipeline' in st.session_state:
                                st.session_state.pipeline.remove_operation(i)
                            st.experimental_rerun()
                
                # Execute pipeline button
                if st.button("Execute Pipeline"):
                    execute_pipeline(data)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("Please upload a data file to begin.")

def configure_missing_value_handler():
    """Configure a missing value handler operation."""
    with st.form("missing_value_form"):
        strategy = st.selectbox(
            "Numeric Strategy",
            ["mean", "median", "most_frequent", "constant"]
        )
        
        fill_value = None
        if strategy == "constant":
            fill_value = st.number_input("Fill Value", value=0)
        
        categorical_strategy = st.selectbox(
            "Categorical Strategy",
            ["most_frequent", "constant"]
        )
        
        exclude_cols = st.multiselect(
            "Exclude Columns",
            st.session_state.data.columns.tolist() if 'data' in st.session_state else []
        )
        
        submitted = st.form_submit_button("Add to Pipeline")
        
        if submitted:
            operation = MissingValueHandler(
                strategy=strategy,
                fill_value=fill_value,
                categorical_strategy=categorical_strategy,
                exclude_cols=exclude_cols
            )
            
            description = f"Strategy: {strategy}, Cat Strategy: {categorical_strategy}"
            if strategy == "constant":
                description += f", Fill Value: {fill_value}"
            
            if 'pipeline' in st.session_state:
                st.session_state.pipeline.add_operation(operation)
            
            st.session_state.pipeline_operations.append({
                "type": "Missing Value Handler",
                "operation": operation,
                "description": description
            })
            
            st.success("Missing Value Handler added to pipeline!")

def configure_outlier_handler():
    """Configure an outlier handler operation."""
    with st.form("outlier_form"):
        method = st.selectbox(
            "Detection Method",
            ["zscore", "iqr"]
        )
        
        threshold = st.number_input(
            "Threshold",
            value=3.0,
            min_value=0.1,
            help="Z-score threshold or IQR multiplier"
        )
        
        strategy = st.selectbox(
            "Handling Strategy",
            ["winsorize", "remove"]
        )
        
        exclude_cols = st.multiselect(
            "Exclude Columns",
            st.session_state.data.columns.tolist() if 'data' in st.session_state else []
        )
        
        submitted = st.form_submit_button("Add to Pipeline")
        
        if submitted:
            operation = OutlierHandler(
                method=method,
                threshold=threshold,
                strategy=strategy,
                exclude_cols=exclude_cols
            )
            
            description = f"Method: {method}, Threshold: {threshold}, Strategy: {strategy}"
            
            if 'pipeline' in st.session_state:
                st.session_state.pipeline.add_operation(operation)
            
            st.session_state.pipeline_operations.append({
                "type": "Outlier Handler",
                "operation": operation,
                "description": description
            })
            
            st.success("Outlier Handler added to pipeline!")

def configure_feature_scaler():
    """Configure a feature scaler operation."""
    with st.form("scaler_form"):
        method = st.selectbox(
            "Scaling Method",
            ["minmax", "standard", "robust"]
        )
        
        feature_range = None
        if method == "minmax":
            min_val = st.number_input("Min Value", value=0.0)
            max_val = st.number_input("Max Value", value=1.0)
            feature_range = (min_val, max_val)
        
        exclude_cols = st.multiselect(
            "Exclude Columns",
            st.session_state.data.columns.tolist() if 'data' in st.session_state else []
        )
        
        submitted = st.form_submit_button("Add to Pipeline")
        
        if submitted:
            operation = FeatureScaler(
                method=method,
                feature_range=feature_range if method == "minmax" else (0, 1),
                exclude_cols=exclude_cols
            )
            
            description = f"Method: {method}"
            if method == "minmax":
                description += f", Range: {feature_range}"
            
            if 'pipeline' in st.session_state:
                st.session_state.pipeline.add_operation(operation)
            
            st.session_state.pipeline_operations.append({
                "type": "Feature Scaler",
                "operation": operation,
                "description": description
            })
            
            st.success("Feature Scaler added to pipeline!")

def configure_category_encoder():
    """Configure a category encoder operation."""
    with st.form("encoder_form"):
        method = st.selectbox(
            "Encoding Method",
            ["label", "onehot"]
        )
        
        exclude_cols = st.multiselect(
            "Exclude Columns",
            st.session_state.data.columns.tolist() if 'data' in st.session_state else []
        )
        
        submitted = st.form_submit_button("Add to Pipeline")
        
        if submitted:
            operation = CategoryEncoder(
                method=method,
                exclude_cols=exclude_cols
            )
            
            description = f"Method: {method}"
            
            if 'pipeline' in st.session_state:
                st.session_state.pipeline.add_operation(operation)
            
            st.session_state.pipeline_operations.append({
                "type": "Category Encoder",
                "operation": operation,
                "description": description
            })
            
            st.success("Category Encoder added to pipeline!")

def configure_feature_selector():
    """Configure a feature selector operation."""
    with st.form("selector_form"):
        method = st.selectbox(
            "Selection Method",
            ["variance", "kbest", "evolutionary"]
        )
        
        threshold = None
        k = None
        target_col = None
        use_evolutionary = False
        ec_algorithm = None
        
        if method == "variance":
            threshold = st.number_input("Variance Threshold", value=0.0, min_value=0.0)
        elif method == "kbest":
            k = st.number_input("Number of Features to Select", value=10, min_value=1)
            target_col = st.selectbox(
                "Target Column",
                st.session_state.data.columns.tolist() if 'data' in st.session_state else []
            )
        elif method == "evolutionary":
            use_evolutionary = True
            ec_algorithm = st.selectbox(
                "Evolutionary Algorithm",
                ["aco", "de", "gwo"]
            )
            target_col = st.selectbox(
                "Target Column",
                st.session_state.data.columns.tolist() if 'data' in st.session_state else []
            )
        
        exclude_cols = st.multiselect(
            "Exclude Columns",
            st.session_state.data.columns.tolist() if 'data' in st.session_state else []
        )
        
        submitted = st.form_submit_button("Add to Pipeline")
        
        if submitted:
            operation = FeatureSelector(
                method=method,
                threshold=threshold if method == "variance" else 0.0,
                k=k if method == "kbest" else None,
                target_col=target_col if method in ["kbest", "evolutionary"] else None,
                exclude_cols=exclude_cols,
                use_evolutionary=use_evolutionary,
                ec_algorithm=ec_algorithm if use_evolutionary else "aco"
            )
            
            description = f"Method: {method}"
            if method == "variance":
                description += f", Threshold: {threshold}"
            elif method == "kbest":
                description += f", K: {k}, Target: {target_col}"
            elif method == "evolutionary":
                description += f", Algorithm: {ec_algorithm}, Target: {target_col}"
            
            if 'pipeline' in st.session_state:
                st.session_state.pipeline.add_operation(operation)
            
            st.session_state.pipeline_operations.append({
                "type": "Feature Selector",
                "operation": operation,
                "description": description
            })
            
            st.success("Feature Selector added to pipeline!")

def configure_time_series_processor():
    """Configure a time series processor operation."""
    with st.form("time_series_form"):
        time_col = st.selectbox(
            "Time Column",
            st.session_state.data.columns.tolist() if 'data' in st.session_state else []
        )
        
        resample_freq = st.text_input(
            "Resample Frequency",
            value="",
            help="e.g., 'D' for daily, 'H' for hourly, '15min' for 15 minutes. Leave blank for no resampling."
        )
        
        lag_features_input = st.text_input(
            "Lag Features",
            value="",
            help="Comma-separated list of lag periods, e.g., '1,2,3'"
        )
        
        rolling_windows_input = st.text_input(
            "Rolling Windows",
            value="",
            help="Comma-separated list of window sizes, e.g., '3,7,14'"
        )
        
        exclude_cols = st.multiselect(
            "Exclude Columns",
            st.session_state.data.columns.tolist() if 'data' in st.session_state else []
        )
        
        submitted = st.form_submit_button("Add to Pipeline")
        
        if submitted:
            lag_features = [int(x.strip()) for x in lag_features_input.split(",") if x.strip()] if lag_features_input else None
            rolling_windows = [int(x.strip()) for x in rolling_windows_input.split(",") if x.strip()] if rolling_windows_input else None
            
            operation = TimeSeriesProcessor(
                time_col=time_col,
                resample_freq=resample_freq if resample_freq else None,
                lag_features=lag_features,
                rolling_windows=rolling_windows,
                exclude_cols=exclude_cols
            )
            
            description = f"Time Col: {time_col}"
            if resample_freq:
                description += f", Resample: {resample_freq}"
            if lag_features:
                description += f", Lags: {lag_features}"
            if rolling_windows:
                description += f", Windows: {rolling_windows}"
            
            if 'pipeline' in st.session_state:
                st.session_state.pipeline.add_operation(operation)
            
            st.session_state.pipeline_operations.append({
                "type": "Time Series Processor",
                "operation": operation,
                "description": description
            })
            
            st.success("Time Series Processor added to pipeline!")

def execute_pipeline(data):
    """Execute the preprocessing pipeline on the data."""
    try:
        # Create a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Execute the pipeline
        pipeline = st.session_state.pipeline
        result = pipeline.fit_transform(data_copy)
        
        # Store the result in session state
        st.session_state.pipeline_result = result
        
        # Display the result
        st.subheader("Pipeline Result")
        st.dataframe(result.head())
        
        # Display quality metrics
        quality_metrics = pipeline.get_quality_metrics()
        st.subheader("Quality Metrics")
        
        for op_name, metrics in quality_metrics.items():
            with st.expander(f"{op_name} Metrics"):
                metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                st.dataframe(metrics_df)
        
        # Save pipeline configuration
        save_dir = Path("pipeline_configs")
        save_dir.mkdir(exist_ok=True)
        
        pipeline_name = st.text_input("Pipeline Name", value=pipeline.name)
        if st.button("Save Pipeline Configuration"):
            pipeline.name = pipeline_name
            pipeline.save_config(str(save_dir / f"{pipeline_name}.json"))
            st.success(f"Pipeline configuration saved as {pipeline_name}.json")
    
    except Exception as e:
        st.error(f"Error executing pipeline: {str(e)}")

def render_data_quality_visualization():
    """Render the data quality visualization component."""
    st.subheader("Data Quality Visualization")
    
    if 'data' not in st.session_state:
        st.info("Please upload data in the Pipeline Builder tab first.")
        return
    
    data = st.session_state.data
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs([
        "Missing Values", 
        "Outliers", 
        "Distributions", 
        "Correlations",
        "Time Series"
    ])
    
    # Missing Values Tab
    with viz_tabs[0]:
        st.write("### Missing Values Analysis")
        
        # Calculate missing values
        missing = data.isnull().sum()
        missing_percent = missing / len(data) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percent': missing_percent
        }).sort_values('Percent', ascending=False)
        
        # Filter to only show columns with missing values
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_df) > 0:
            st.bar_chart(missing_df['Percent'])
            st.dataframe(missing_df)
        else:
            st.success("No missing values found in the dataset!")
    
    # Outliers Tab
    with viz_tabs[1]:
        st.write("### Outlier Analysis")
        
        # Select column for outlier analysis
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for outlier analysis.")
            return
        
        selected_col = st.selectbox("Select Column for Outlier Analysis", numeric_cols)
        
        # Calculate outliers using IQR method
        Q1 = data[selected_col].quantile(0.25)
        Q3 = data[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]
        
        # Display boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        data.boxplot(column=[selected_col], ax=ax)
        st.pyplot(fig)
        
        # Display outlier information
        st.write(f"**Outlier Boundaries:** Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")
        st.write(f"**Number of Outliers:** {len(outliers)}")
        
        if len(outliers) > 0:
            st.dataframe(outliers[[selected_col]])
    
    # Distributions Tab
    with viz_tabs[2]:
        st.write("### Distribution Analysis")
        
        # Select column for distribution analysis
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for distribution analysis.")
            return
        
        selected_col = st.selectbox("Select Column for Distribution Analysis", numeric_cols)
        
        # Display histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        data[selected_col].hist(bins=30, ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)
        
        # Display statistics
        stats = data[selected_col].describe()
        st.write("**Summary Statistics:**")
        st.dataframe(pd.DataFrame(stats).T)
    
    # Correlations Tab
    with viz_tabs[3]:
        st.write("### Correlation Analysis")
        
        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.shape[1] < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
            return
        
        corr_matrix = numeric_data.corr()
        
        # Display heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Display top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_pairs.append({
                    'Feature 1': col1,
                    'Feature 2': col2,
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
        st.write("**Top Feature Correlations:**")
        st.dataframe(corr_df)
    
    # Time Series Tab
    with viz_tabs[4]:
        st.write("### Time Series Analysis")
        
        # Check if there are datetime columns
        datetime_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col])
                datetime_cols.append(col)
            except:
                continue
        
        if not datetime_cols:
            st.warning("No datetime columns found for time series analysis.")
            return
        
        time_col = st.selectbox("Select Time Column", datetime_cols)
        
        # Convert to datetime
        data_ts = data.copy()
        data_ts[time_col] = pd.to_datetime(data_ts[time_col])
        
        # Select feature to analyze
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for time series analysis.")
            return
        
        selected_feature = st.selectbox("Select Feature for Time Series Analysis", numeric_cols)
        
        # Sort by time
        data_ts = data_ts.sort_values(time_col)
        
        # Plot time series
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data_ts[time_col], data_ts[selected_feature])
        ax.set_title(f"{selected_feature} over Time")
        ax.set_xlabel(time_col)
        ax.set_ylabel(selected_feature)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Resample options
        st.write("### Resampling Analysis")
        resample_freq = st.selectbox(
            "Resample Frequency",
            ["D", "W", "M", "Q", "Y"],
            format_func=lambda x: {
                "D": "Daily",
                "W": "Weekly",
                "M": "Monthly",
                "Q": "Quarterly",
                "Y": "Yearly"
            }[x]
        )
        
        try:
            # Resample data
            data_resampled = data_ts.set_index(time_col).resample(resample_freq)
            data_resampled = data_resampled[selected_feature].mean().reset_index()
            
            # Plot resampled data
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data_resampled[time_col], data_resampled[selected_feature])
            ax.set_title(f"{selected_feature} over Time (Resampled to {resample_freq})")
            ax.set_xlabel(time_col)
            ax.set_ylabel(f"Mean {selected_feature}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error resampling data: {str(e)}")

def render_template_management():
    """Render the template management component with enhanced features."""
    st.subheader("Template Management System")
    
    # Create directory for pipeline templates if it doesn't exist
    template_dir = Path("pipeline_templates")
    template_dir.mkdir(exist_ok=True)
    
    # Create categories directory if it doesn't exist
    categories_dir = template_dir / "categories"
    categories_dir.mkdir(exist_ok=True)
    
    # Get list of available templates
    templates = [f for f in template_dir.glob("*.json") if f.is_file()]
    category_templates = []
    for category_dir in categories_dir.glob("*/"):
        category_templates.extend([(category_dir.name, f) for f in category_dir.glob("*.json") if f.is_file()])
    
    # Create tabs for template operations
    template_tabs = st.tabs(["Browse Templates", "Save Template", "Share Templates", "Import Template"])
    
    # Browse Templates Tab
    with template_tabs[0]:
        st.write("### Browse Available Templates")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_option = st.radio("Filter By", ["All Templates", "Categories", "Tags"])
        
        with col2:
            search_term = st.text_input("Search Templates", "")
        
        # Display templates based on filter
        if filter_option == "All Templates":
            display_all_templates(templates, category_templates, search_term)
        elif filter_option == "Categories":
            display_templates_by_category(category_templates, search_term)
        elif filter_option == "Tags":
            display_templates_by_tags(templates, category_templates, search_term)
    
    # Save Template Tab
    with template_tabs[1]:
        if 'pipeline' in st.session_state and len(st.session_state.pipeline.operations) > 0:
            st.write("### Save Current Pipeline as Template")
            
            # Basic template info
            template_name = st.text_input("Template Name", value=st.session_state.pipeline.name)
            template_description = st.text_area("Template Description", value="")
            
            # Advanced template options
            with st.expander("Advanced Options", expanded=False):
                # Category selection
                categories = [d.name for d in categories_dir.glob("*/") if d.is_dir()]
                categories.append("Create New Category")
                selected_category = st.selectbox("Category", ["None"] + categories)
                
                new_category = None
                if selected_category == "Create New Category":
                    new_category = st.text_input("New Category Name")
                
                # Tags
                tags = st.text_input("Tags (comma-separated)", value="")
                
                # Version
                version = st.text_input("Version", value="1.0.0")
                
                # Compatibility
                compatibility = st.multiselect("Compatible With", 
                                             ["Clinical Data", "Time Series", "Categorical", "Numerical"],
                                             default=["Numerical"])
            
            if st.button("Save as Template"):
                try:
                    # Update pipeline name
                    st.session_state.pipeline.name = template_name
                    
                    # Save pipeline configuration
                    pipeline_config = st.session_state.pipeline.get_params()
                    pipeline_config['description'] = template_description
                    
                    with open(str(template_dir / f"{template_name}.json"), 'w') as f:
                        json.dump(pipeline_config, f, indent=2)
                    
                    st.success(f"Pipeline saved as template '{template_name}'!")
                
                except Exception as e:
                    st.error(f"Error saving template: {str(e)}")
        else:
            st.info("Create a pipeline in the Pipeline Builder tab first.")
    
    # Share Template Tab
    with template_tabs[2]:
        if templates:
            template_names = [t.stem for t in templates]
            selected_template = st.selectbox("Select Template to Share", template_names)
            
            if st.button("Generate Shareable Configuration"):
                try:
                    # Load the template
                    with open(str(template_dir / f"{selected_template}.json"), 'r') as f:
                        template_config = json.load(f)
                    
                    # Display the configuration
                    st.subheader("Template Configuration")
                    st.json(template_config)
                    
                    # Provide download link
                    st.download_button(
                        label="Download Template Configuration",
                        data=json.dumps(template_config, indent=2),
                        file_name=f"{selected_template}.json",
                        mime="application/json"
                    )
                
                except Exception as e:
                    st.error(f"Error generating shareable configuration: {str(e)}")
        else:
            st.info("No templates available to share.")

def display_all_templates(templates, category_templates, search_term=""):
    """Display all available templates with filtering by search term."""
    all_templates = templates.copy()
    all_templates.extend([t[1] for t in category_templates])
    
    if not all_templates:
        st.info("No templates available. Create and save a pipeline in the Save Template tab.")
        return
    
    filtered_templates = []
    for template in all_templates:
        # Load template metadata
        try:
            with open(template, 'r') as f:
                data = json.load(f)
            
            # Check if template matches search term
            if search_term and search_term.lower() not in template.stem.lower() and search_term.lower() not in data.get('description', '').lower():
                continue
            
            filtered_templates.append((template, data))
        except Exception as e:
            st.warning(f"Could not load template {template.name}: {str(e)}")
    
    display_template_list(filtered_templates)

def display_templates_by_category(category_templates, search_term=""):
    """Display templates organized by category with filtering by search term."""
    if not category_templates:
        st.info("No categorized templates available.")
        return
    
    # Group templates by category
    categories = {}
    for category, template in category_templates:
        if category not in categories:
            categories[category] = []
        categories[category].append(template)
    
    # Display templates by category
    for category, templates in categories.items():
        with st.expander(f"{category} ({len(templates)})", expanded=True):
            filtered_templates = []
            for template in templates:
                # Load template metadata
                try:
                    with open(template, 'r') as f:
                        data = json.load(f)
                    
                    # Check if template matches search term
                    if search_term and search_term.lower() not in template.stem.lower() and search_term.lower() not in data.get('description', '').lower():
                        continue
                    
                    filtered_templates.append((template, data))
                except Exception as e:
                    st.warning(f"Could not load template {template.name}: {str(e)}")
            
            if filtered_templates:
                display_template_list(filtered_templates)
            else:
                st.info(f"No templates matching '{search_term}' in this category.")

def display_templates_by_tags(templates, category_templates, search_term=""):
    """Display templates organized by tags with filtering by search term."""
    all_templates = templates.copy()
    all_templates.extend([t[1] for t in category_templates])
    
    if not all_templates:
        st.info("No templates available.")
        return
    
    # Group templates by tags
    tag_groups = {}
    for template in all_templates:
        # Load template metadata
        try:
            with open(template, 'r') as f:
                data = json.load(f)
            
            # Check if template matches search term
            if search_term and search_term.lower() not in template.stem.lower() and search_term.lower() not in data.get('description', '').lower():
                continue
            
            # Get tags
            tags = data.get('tags', [])
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',')]
            
            if not tags:
                tags = ['Untagged']
            
            # Add to tag groups
            for tag in tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append((template, data))
        except Exception as e:
            st.warning(f"Could not load template {template.name}: {str(e)}")
    
    # Display templates by tag
    for tag, templates in tag_groups.items():
        with st.expander(f"{tag} ({len(templates)})", expanded=True):
            display_template_list(templates)

def display_template_list(templates):
    """Display a list of templates with options to view details, load, or delete."""
    for template_path, template_data in templates:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            template_name = template_data.get('name', template_path.stem)
            st.write(f"**{template_name}**")
            description = template_data.get('description', 'No description available')
            st.write(f"<small>{description[:100]}{'...' if len(description) > 100 else ''}</small>", unsafe_allow_html=True)
        
        with col2:
            if st.button("Details", key=f"details_{template_path.stem}"):
                display_template_details(template_path, template_data)
        
        with col3:
            if st.button("Load", key=f"load_{template_path.stem}"):
                load_template(template_path)
        
        with col4:
            if st.button("Delete", key=f"delete_{template_path.stem}"):
                if delete_template(template_path):
                    st.experimental_rerun()
        
        st.markdown("---")

def display_template_details(template_path, template_data):
    """Display detailed information about a template."""
    with st.expander("Template Details", expanded=True):
        st.write(f"### {template_data.get('name', template_path.stem)}")
        
        # Basic info
        st.write(f"**Description:** {template_data.get('description', 'No description available')}")
        st.write(f"**Version:** {template_data.get('version', '1.0.0')}")
        st.write(f"**Created:** {template_data.get('created_at', 'Unknown')}")
        
        # Tags
        tags = template_data.get('tags', [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',')]
        
        if tags:
            st.write("**Tags:**")
            st.write(", ".join(tags))
        
        # Compatibility
        compatibility = template_data.get('compatibility', [])
        if compatibility:
            st.write("**Compatible With:**")
            st.write(", ".join(compatibility))
        
        # Pipeline operations
        operations = template_data.get('operations', [])
        if operations:
            st.write("**Pipeline Operations:**")
            for i, op in enumerate(operations):
                st.write(f"{i+1}. {op.get('type', 'Unknown')}")
                params = op.get('params', {})
                if params:
                    st.write(f"   Parameters: {', '.join([f'{k}={v}' for k, v in params.items()])}")

def load_template(template_path):
    """Load a template into the current pipeline."""
    try:
        # Load the template
        pipeline = PreprocessingPipeline.load_config(str(template_path))
        
        # Update session state
        st.session_state.pipeline = pipeline
        
        # Update pipeline operations
        st.session_state.pipeline_operations = []
        for i, op in enumerate(pipeline.operations):
            op_type = op.__class__.__name__
            params = op.get_params()
            
            description = f"Type: {op_type}"
            for key, value in params.items():
                if key not in ['exclude_cols']:
                    description += f", {key}: {value}"
            
            st.session_state.pipeline_operations.append({
                "type": op_type,
                "operation": op,
                "description": description
            })
        
        st.success(f"Template '{template_path.stem}' loaded successfully!")
        st.info("Go to the Pipeline Builder tab to see the loaded pipeline.")
        return True
    
    except Exception as e:
        st.error(f"Error loading template: {str(e)}")
        return False

def delete_template(template_path):
    """Delete a template file."""
    try:
        # Confirm deletion
        if st.session_state.get(f"confirm_delete_{template_path.stem}", False):
            # Delete the file
            template_path.unlink()
            st.success(f"Template '{template_path.stem}' deleted successfully!")
            return True
        else:
            st.warning(f"Are you sure you want to delete '{template_path.stem}'?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete", key=f"confirm_yes_{template_path.stem}"):
                    st.session_state[f"confirm_delete_{template_path.stem}"] = True
                    return delete_template(template_path)
            with col2:
                if st.button("No, cancel", key=f"confirm_no_{template_path.stem}"):
                    st.session_state[f"confirm_delete_{template_path.stem}"] = False
            return False
    
    except Exception as e:
        st.error(f"Error deleting template: {str(e)}")
        return False

def render_data_configuration_ui():
    """Wrapper function for render_data_configuration to maintain compatibility with the benchmark dashboard."""
    render_data_configuration()
    
    
# Function to convert between old and new preprocessing formats
def convert_to_advanced_pipeline(basic_pipeline):
    """Convert a basic preprocessing pipeline to the advanced preprocessing manager format.
    
    Args:
        basic_pipeline: A PreprocessingPipeline instance
        
    Returns:
        A PreprocessingManager instance with equivalent operations
    """
    # Create a new preprocessing manager
    manager = PreprocessingManager()
    
    # Initialize configuration
    config = {
        'pipeline_name': basic_pipeline.name,
        'operations': {},
        'advanced_operations': {},
        'domain_operations': {}
    }
    
    # Convert operations
    for op in basic_pipeline.operations:
        if isinstance(op, MissingValueHandler):
            config['operations']['missing_value_handler'] = {
                'include': True,
                'params': {
                    'strategy': op.strategy,
                    'fill_value': op.fill_value,
                    'categorical_strategy': op.categorical_strategy,
                    'exclude_cols': op.exclude_cols
                }
            }
        elif isinstance(op, OutlierHandler):
            config['operations']['outlier_handler'] = {
                'include': True,
                'params': {
                    'method': op.method,
                    'threshold': op.threshold,
                    'strategy': op.strategy,
                    'exclude_cols': op.exclude_cols
                }
            }
        elif isinstance(op, FeatureScaler):
            config['operations']['feature_scaler'] = {
                'include': True,
                'params': {
                    'method': op.method,
                    'exclude_cols': op.exclude_cols
                }
            }
        elif isinstance(op, CategoryEncoder):
            config['operations']['category_encoder'] = {
                'include': True,
                'params': {
                    'method': op.method,
                    'exclude_cols': op.exclude_cols
                }
            }
        elif isinstance(op, FeatureSelector):
            config['operations']['feature_selector'] = {
                'include': True,
                'params': {
                    'method': op.method,
                    'threshold': getattr(op, 'threshold', None),
                    'k': getattr(op, 'k', None),
                    'exclude_cols': op.exclude_cols
                }
            }
    
    # Update manager configuration
    manager.update_config(config)
    
    return manager


def convert_to_basic_pipeline(manager):
    """Convert an advanced preprocessing manager to a basic preprocessing pipeline.
    
    Args:
        manager: A PreprocessingManager instance
        
    Returns:
        A PreprocessingPipeline instance with equivalent operations
    """
    # Get configuration
    config = manager.get_config()
    
    # Create a new preprocessing pipeline
    pipeline = PreprocessingPipeline(name=config.get('pipeline_name', 'Converted Pipeline'))
    
    # Convert operations
    operations = config.get('operations', {})
    
    # Missing value handler
    missing_config = operations.get('missing_value_handler', {})
    if missing_config.get('include', False):
        params = missing_config.get('params', {})
        pipeline.add_operation(MissingValueHandler(
            strategy=params.get('strategy', 'mean'),
            fill_value=params.get('fill_value', None),
            categorical_strategy=params.get('categorical_strategy', 'most_frequent'),
            exclude_cols=params.get('exclude_cols', [])
        ))
    
    # Outlier handler
    outlier_config = operations.get('outlier_handler', {})
    if outlier_config.get('include', False):
        params = outlier_config.get('params', {})
        pipeline.add_operation(OutlierHandler(
            method=params.get('method', 'zscore'),
            threshold=params.get('threshold', 3.0),
            strategy=params.get('strategy', 'winsorize'),
            exclude_cols=params.get('exclude_cols', [])
        ))
    
    # Feature scaler
    scaler_config = operations.get('feature_scaler', {})
    if scaler_config.get('include', False):
        params = scaler_config.get('params', {})
        pipeline.add_operation(FeatureScaler(
            method=params.get('method', 'standard'),
            exclude_cols=params.get('exclude_cols', [])
        ))
    
    # Category encoder
    encoder_config = operations.get('category_encoder', {})
    if encoder_config.get('include', False):
        params = encoder_config.get('params', {})
        pipeline.add_operation(CategoryEncoder(
            method=params.get('method', 'onehot'),
            exclude_cols=params.get('exclude_cols', [])
        ))
    
    # Feature selector
    selector_config = operations.get('feature_selector', {})
    if selector_config.get('include', False):
        params = selector_config.get('params', {})
        pipeline.add_operation(FeatureSelector(
            method=params.get('method', 'variance'),
            threshold=params.get('threshold', 0.01),
            k=params.get('k', 10),
            exclude_cols=params.get('exclude_cols', [])
        ))
    
    return pipeline


