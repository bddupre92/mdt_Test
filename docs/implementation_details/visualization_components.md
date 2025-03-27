# MoE Interactive Visualization Components

## HTML Report Structure and Visualization

The MoE framework generates comprehensive interactive HTML reports that visualize various aspects of the validation results. This document details how these visualizations work and their role in the system.

### Report Structure

The interactive HTML report (`real_data_report_*.html`) is structured with the following key components:

1. **HTML Framework**: Based on a modern HTML5 template with responsive design
2. **CSS Styling**: Uses Bootstrap and custom CSS for responsive layouts
3. **JavaScript Functionality**: Leverages Plotly.js for interactive visualizations
4. **Tabbed Interface**: Organizes visualizations into logical sections

### Key Visualization Components

#### 1. Data Quality Visualizations

**Implementation Files**:
- `tests/moe_interactive_report.py`: Main report generation
- `tests/clinical_metrics_report.py`: Clinical data metrics
- Template section in report HTML structure

**Visualization Types**:
- Data completeness bar charts
- Data distribution histograms
- Data quality issue summaries
- Time-series data validation plots

**User Interactions**:
- Hover for detailed metrics
- Toggle between different data quality aspects
- Drill down into specific data features

#### 2. Feature Importance Visualizations

**Implementation Files**:
- `explainability/shap_explainer.py`: SHAP calculation
- `direct_validation_test.py`: Integration with validation
- HTML/JS rendering in reports

**Visualization Types**:
- SHAP summary plots
- Feature importance bar charts
- Feature correlation heatmaps
- Feature contribution waterfall charts

**User Interactions**:
- Sort features by importance
- Filter features by category
- View detailed feature impact explanations

#### 3. Model Performance Visualizations

**Implementation Files**:
- `tests/model_evaluation_report.py`: Performance metrics
- `tests/evolutionary_performance_report.py`: Learning curves
- JavaScript rendering in reports

**Visualization Types**:
- Performance metric charts (accuracy, precision, recall, F1)
- ROC and precision-recall curves
- Confusion matrices
- Learning and validation curves

**User Interactions**:
- Toggle between different metrics
- Compare performance across different models
- View detailed performance statistics

#### 4. Drift Analysis Visualizations

**Implementation Files**:
- `tests/drift_performance_report.py`: Drift detection
- HTML/JS rendering in reports

**Visualization Types**:
- Drift pattern visualizations
- Distribution shift charts
- Time-series drift tracking
- Feature stability monitors

**User Interactions**:
- View drift patterns over time
- Compare drift across different features
- Analyze drift severity and impact

### Technical Implementation

The visualization system is implemented through a multi-layered approach:

1. **Data Preparation Layer**:
   - Raw data and validation results are processed into visualization-ready formats
   - Statistical calculations are performed to generate summary metrics
   - Data is structured for efficient rendering

2. **Visualization Generation Layer**:
   - Plotly.js is used to create interactive charts and graphs
   - Custom JavaScript functions handle user interactions
   - Templates define the basic structure of each visualization

3. **Integration Layer**:
   - Visualizations are embedded into the HTML report template
   - Interactive elements are connected to the underlying data
   - Navigation system allows moving between different visualizations

### Example: SHAP Visualization Workflow

The SHAP visualization process follows these steps:

1. SHAP values are calculated in `explainability/shap_explainer.py`
2. Summary data is saved to JSON format
3. The report generator reads this data and creates visualization specifications
4. Plotly.js renders the visualizations in the report
5. The final interactive report allows exploration of feature importance

### Extending Visualizations

The visualization system is designed to be extensible:

1. **Adding New Visualizations**:
   - Create a new report module (e.g., `new_feature_report.py`)
   - Implement the visualization generation functions
   - Add the module to the main report generator

2. **Customizing Existing Visualizations**:
   - Modify chart configuration in the report generator
   - Adjust JavaScript interaction handlers
   - Update CSS styling for visual appearance

3. **Integration with External Tools**:
   - Export visualization data for use in BI tools
   - Generate standalone visualization files
   - Create API endpoints for dynamic visualization updates

## Running Visualizations with MoE Validation

### Generating Interactive Reports

The MoE validation framework can generate comprehensive interactive HTML reports with visualizations through several methods:

#### 1. Synthetic Data Visualization

To generate visualizations using synthetic data:

```bash
python -m tests.moe_validation_runner --drift_type sudden --output_dir ./output/moe_validation --interactive
```

This command will:
- Generate synthetic data with sudden drift patterns
- Run the MoE validation framework
- Create an interactive HTML report with visualizations in the specified output directory

#### 2. Real Data Visualization

To generate visualizations using real clinical data:

```bash
python -m tests.test_real_data_validation
```

Or with more control over the process:

```bash
python -m cli.real_data_commands --clinical_data ./data/clinical_data.csv \
  --data_format csv --target_column migraine_severity \
  --output_dir ./output/real_data_validation --synthetic_compare
```

### Visualization Components and Files

The visualization system is composed of several key files:

| File | Purpose |
|------|--------|
| `tests/moe_interactive_report.py` | Main report generator that integrates all visualization components |
| `tests/enhanced_data_report.py` | Generates visualizations for enhanced data features |
| `tests/expert_performance_report.py` | Creates expert model performance visualizations |
| `tests/model_evaluation_report.py` | Produces model evaluation metric visualizations |
| `tests/clinical_metrics_report.py` | Generates clinical data metric visualizations |
| `tests/real_data_validation_report.py` | Creates visualizations specific to real data validation |

### Integrating Real Data

To integrate real clinical data into the visualization system:

1. **Prepare Your Data**:
   - Ensure your data is in CSV, JSON, or Excel format
   - Verify that required columns are present (patient_id, timestamp, features, target)
   - Clean and preprocess data as needed

2. **Run the Integration Command**:
   ```bash
   python -m cli.real_data_commands --clinical_data [path_to_data] \
     --data_format [format] --target_column [target] \
     --output_dir [output_directory]
   ```

3. **Enable Synthetic Comparison** (optional):
   - Add the `--synthetic_compare` flag to compare real data with synthetic data
   - Specify drift type with `--drift_type [type]` for synthetic comparison

4. **View the Generated Report**:
   - Open the HTML file generated in the output directory
   - Navigate through the tabs to view different visualization sections

### Customizing Visualizations

To customize the visualizations in the reports:

1. **Modify Chart Parameters**:
   - Edit the relevant report module (e.g., `enhanced_data_report.py`)
   - Adjust chart parameters, colors, and layouts

2. **Add New Visualization Types**:
   - Create a new function in the appropriate report module
   - Add the function call to the report generation sequence

3. **Change Report Structure**:
   - Modify the tab structure in `moe_interactive_report.py`
   - Adjust the HTML template structure as needed

### Troubleshooting Visualizations

If you encounter issues with visualizations:

1. **Check Browser Console**: Most visualization issues can be diagnosed in the browser's developer console
2. **Verify Data Format**: Ensure your data is in the expected format for each visualization
3. **Update Plotly.js**: Some visualizations may require specific versions of Plotly.js
4. **Check for Syntax Errors**: Ensure JavaScript code in HTML templates is correctly formatted

## Dashboard Integration

The interactive HTML reports are designed to serve as a foundation for a comprehensive dashboard:

1. **Component Reusability**: All visualizations can be extracted as modules
2. **Data API Integration**: Visualization components connect to data through a defined API
3. **Real-time Updates**: Structure supports dynamic data updates
4. **User Customization**: Layout and visualization preferences can be saved

### Dashboard Implementation Roadmap

To implement a full dashboard based on these visualizations:

1. **Web Application Framework**: Use Flask or Django to create a server
2. **Data API Layer**: Create REST endpoints for accessing validation data
3. **Authentication System**: Add user management and authentication
4. **Dashboard Layout**: Design a customizable dashboard layout
5. **Visualization Components**: Port existing visualizations to dashboard widgets
