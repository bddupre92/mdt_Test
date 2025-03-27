# AI Dashboard Creation Prompt for Migraine Prediction MoE Framework

## Context and Task Description

I need you to create a comprehensive dashboard for a Mixture-of-Experts (MoE) framework that predicts migraines using multiple data modalities. The dashboard should allow importing clinical data, preprocessing it, running the MoE model with SHAP explainability, and visualizing results through interactive reports.

## System Architecture

The MoE framework consists of these key components:

1. **Data Preprocessing**: Handles validation, datetime conversion, and feature engineering
2. **Expert Models**: Domain-specific models for physiological, environmental, behavioral, and medication data
3. **Gating Network**: Weights expert predictions based on input characteristics
4. **SHAP Explainability**: Analyzes feature importance for model interpretability
5. **Interactive Reporting**: Visualizes performance metrics, feature importance, and data quality

## Required Dashboard Capabilities

Please create a full-stack dashboard application with these core features:

### 1. Data Management
- Upload/import clinical data (CSV format)
- Generate synthetic patient data with configurable drift patterns
- Validate data quality and completeness
- Visualize data distributions and quality metrics

### 2. Model Configuration and Execution
- Configure expert models and gating parameters
- Train MoE model on imported data
- Generate predictions with confidence scores
- Perform SHAP analysis for explainability

### 3. Interactive Visualizations
- Model performance metrics (accuracy, precision, recall, F1)
- Feature importance visualization from SHAP
- Patient-specific prediction explanations
- Drift detection and monitoring visualizations

### 4. User Experience
- Clean, modern interface with responsive design
- Tabbed navigation between different sections
- Interactive charts with hover tooltips
- Downloadable reports and visualizations

## Technical Requirements

The dashboard should be implemented using:

- **Backend**: Python-based web framework (Flask or Django)
- **Frontend**: Modern JavaScript framework with Plotly.js for visualizations
- **Data Processing**: Integration with existing preprocessing pipelines
- **Model Integration**: Connection to MoE model implementation
- **Deployment**: Containerized for easy deployment

## Implementation Reference Documents

Please refer to these specific documents in the implementation:

1. **System Architecture**: Detailed in `architecture_overview.md`, including component relationships and data flow
2. **Visualization Components**: Described in `visualization_components.md`, including report structure and visualization types
3. **AI Dashboard Guide**: Implementation blueprint available in `ai_dashboard_guide.md`, with code examples
4. **Clinical Impact**: Performance metrics and clinical benefits documented in `clinical_impact_summary.md`
5. **Future Enhancements**: Roadmap in `future_enhancements.md`, particularly sections 3.2 (Advanced Visualization) and 6.1 (Production Monitoring)

## Key Integration Points

The dashboard must integrate with these specific components:

1. **Data Preprocessing**: Connect with `clinical_data_validator.py` for data validation
2. **MoE Model**: Interface with `moe/moe_model.py` for prediction generation
3. **SHAP Analysis**: Integrate with `explainability/shap_explainer.py` for feature importance
4. **Interactive Reporting**: Leverage components from `tests/moe_interactive_report.py`

## Dashboard Layout

The dashboard should include these main sections:

1. **Home/Overview**: Summary metrics and system status
2. **Data Management**: Data import, validation, and quality visualization
3. **Model Configuration**: Expert model and gating network setup
4. **Prediction Analysis**: Patient-level predictions with explanations
5. **Performance Monitoring**: Model metrics and drift detection

## Comprehensive Visualization Specifications

### 1. Data Quality and Validation Visualizations

#### 1.1 Data Completeness Visualization
- **Chart Type**: Interactive stacked bar chart
- **Data Source**: `clinical_data_validator.py` completeness metrics
- **Implementation**: 
  ```javascript
  function createCompletenessChart(data) {
    const layout = {
      barmode: 'stack',
      title: 'Data Completeness by Feature',
      xaxis: {title: 'Feature'},
      yaxis: {title: 'Completeness (%)', range: [0, 100]}
    };
    
    const completenessTrace = {
      x: data.features,
      y: data.completeness_values,
      type: 'bar',
      name: 'Complete',
      marker: {color: '#4CAF50'}
    };
    
    const missingnessTrace = {
      x: data.features,
      y: data.missingness_values,
      type: 'bar',
      name: 'Missing',
      marker: {color: '#F44336'}
    };
    
    Plotly.newPlot('completeness-chart', [completenessTrace, missingnessTrace], layout);
  }
  ```
- **Interactivity**: Hover for exact percentages, click to see details of missing data patterns

#### 1.2 Feature Distribution Comparison
- **Chart Type**: Multiple overlaid histograms with density curves
- **Data Source**: Real vs. synthetic data distributions
- **Implementation**:
  ```javascript
  function createDistributionComparison(feature, realData, syntheticData) {
    const layout = {
      title: `Distribution Comparison: ${feature}`,
      xaxis: {title: feature},
      yaxis: {title: 'Density'},
      barmode: 'overlay',
      opacity: 0.7
    };
    
    const realTrace = {
      x: realData,
      type: 'histogram',
      name: 'Real Data',
      opacity: 0.6,
      histnorm: 'probability density',
      marker: {color: '#3F51B5'}
    };
    
    const syntheticTrace = {
      x: syntheticData,
      type: 'histogram',
      name: 'Synthetic Data',
      opacity: 0.6,
      histnorm: 'probability density',
      marker: {color: '#FF9800'}
    };
    
    Plotly.newPlot('distribution-chart', [realTrace, syntheticTrace], layout);
  }
  ```
- **Interactivity**: Feature selector dropdown, normalization toggle, bin size adjuster

#### 1.3 Data Quality Heatmap
- **Chart Type**: Interactive heatmap
- **Data Source**: Data quality metrics from validator
- **Implementation**:
  ```javascript
  function createQualityHeatmap(qualityMatrix) {
    const layout = {
      title: 'Data Quality Assessment',
      xaxis: {title: 'Quality Dimension'},
      yaxis: {title: 'Feature'},
      annotations: []
    };
    
    // Add text annotations to each cell
    for (let i = 0; i < qualityMatrix.y.length; i++) {
      for (let j = 0; j < qualityMatrix.x.length; j++) {
        const value = qualityMatrix.z[i][j];
        layout.annotations.push({
          x: qualityMatrix.x[j],
          y: qualityMatrix.y[i],
          text: value.toFixed(1),
          font: {color: value < 50 ? 'white' : 'black'},
          showarrow: false
        });
      }
    }
    
    const heatmapTrace = {
      x: qualityMatrix.x,
      y: qualityMatrix.y,
      z: qualityMatrix.z,
      type: 'heatmap',
      colorscale: 'Viridis'
    };
    
    Plotly.newPlot('quality-heatmap', [heatmapTrace], layout);
  }
  ```
- **Interactivity**: Click cells for detailed quality issue descriptions, filter by quality dimension

### 2. SHAP Explainability Visualizations

#### 2.1 SHAP Summary Plot
- **Chart Type**: Horizontal bar chart with color gradient
- **Data Source**: SHAP values from `explainability/shap_explainer.py`
- **Implementation**:
  ```javascript
  function createShapSummaryPlot(shapData) {
    // Sort features by importance (absolute mean SHAP value)
    const sortedFeatures = [...shapData.features].sort((a, b) => {
      return Math.abs(shapData.mean_values[b]) - Math.abs(shapData.mean_values[a]);
    });
    
    const sortedValues = sortedFeatures.map(feature => {
      return shapData.mean_values[shapData.features.indexOf(feature)];
    });
    
    // Generate color gradient based on values
    const colors = sortedValues.map(value => {
      return value >= 0 ? `rgba(235, 52, 52, ${Math.min(Math.abs(value) / Math.max(...sortedValues), 1)})` :
                        `rgba(52, 152, 235, ${Math.min(Math.abs(value) / Math.max(...sortedValues), 1)})`;
    });
    
    const layout = {
      title: 'Feature Importance (SHAP Values)',
      xaxis: {title: 'mean(|SHAP value|)'},
      yaxis: {title: '', automargin: true},
      margin: {l: 150}
    };
    
    const barTrace = {
      y: sortedFeatures,
      x: sortedValues.map(Math.abs),
      type: 'bar',
      orientation: 'h',
      marker: {color: colors}
    };
    
    Plotly.newPlot('shap-summary', [barTrace], layout);
  }
  ```
- **Interactivity**: Hover for exact SHAP values, click feature to see detailed contribution breakdown

#### 2.2 Individual Prediction Waterfall Chart
- **Chart Type**: Waterfall chart
- **Data Source**: Individual instance SHAP values
- **Implementation**:
  ```javascript
  function createWaterfallChart(patientId, baseValue, shapValues, features) {
    // Sort by absolute SHAP value importance for this patient
    const featureIndices = Array.from(features.keys());
    featureIndices.sort((a, b) => Math.abs(shapValues[b]) - Math.abs(shapValues[a]));
    
    // Take top 10 features for readability
    const topFeatures = featureIndices.slice(0, 10).map(i => features[i]);
    const topValues = featureIndices.slice(0, 10).map(i => shapValues[i]);
    
    // Create measures and base value for waterfall
    const measures = topValues.map(val => val >= 0 ? 'relative' : 'relative');
    measures.push('total');
    
    const featureLabels = [...topFeatures, 'Prediction'];
    
    // Calculate cumulative values for waterfall
    let runningTotal = baseValue;
    const values = [];
    
    topValues.forEach(val => {
      values.push(val);
      runningTotal += val;
    });
    
    // Add final prediction value
    values.push(runningTotal);
    
    const layout = {
      title: `Prediction Explanation for Patient #${patientId}`,
      xaxis: {title: ''},
      yaxis: {title: 'Feature Contribution'},
      waterfallgroupgap: 0.5,
      showlegend: false
    };
    
    const waterfallTrace = {
      name: 'Feature Contribution',
      type: 'waterfall',
      orientation: 'v',
      measure: measures,
      x: featureLabels,
      y: values,
      connector: {
        line: {
          color: 'rgb(63, 63, 63)'
        }
      },
      increasing: {marker: {color: '#FF9800'}},
      decreasing: {marker: {color: '#2196F3'}},
      totals: {marker: {color: '#4CAF50'}}
    };
    
    Plotly.newPlot('waterfall-chart', [waterfallTrace], layout);
  }
  ```
- **Interactivity**: Patient selector dropdown, hover for feature value details

#### 2.3 Feature Dependence Plot
- **Chart Type**: Scatter plot with color gradient
- **Data Source**: SHAP values and feature values
- **Implementation**:
  ```javascript
  function createDependencePlot(feature, featureValues, shapValues, interaction = null) {
    const layout = {
      title: `SHAP Dependence Plot: ${feature}`,
      xaxis: {title: feature},
      yaxis: {title: 'SHAP value impact'},
      coloraxis: {colorbar: {title: interaction || 'Feature Value'}}
    };
    
    const scatterTrace = {
      x: featureValues,
      y: shapValues,
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 10,
        color: interaction ? interactionValues : featureValues,
        colorscale: 'Viridis',
        showscale: true
      }
    };
    
    Plotly.newPlot('dependence-plot', [scatterTrace], layout);
  }
  ```
- **Interactivity**: Feature selector dropdown, interaction feature selector

### 3. Model Performance Visualizations

#### 3.1 Expert Model Contribution Visualization
- **Chart Type**: Stacked area chart
- **Data Source**: Expert model weights from gating network
- **Implementation**:
  ```javascript
  function createExpertContributionChart(patients, expertWeights, expertNames) {
    const layout = {
      title: 'Expert Model Contributions by Patient',
      xaxis: {title: 'Patient ID'},
      yaxis: {title: 'Expert Contribution', tickformat: ',.0%'},
      barmode: 'stack',
      hovermode: 'closest'
    };
    
    const traces = [];
    
    // Create a trace for each expert
    for (let i = 0; i < expertNames.length; i++) {
      traces.push({
        x: patients,
        y: expertWeights.map(weights => weights[i]),
        type: 'bar',
        name: expertNames[i],
        hoverinfo: 'name+y'
      });
    }
    
    Plotly.newPlot('expert-contribution-chart', traces, layout);
  }
  ```
- **Interactivity**: Hover for exact contribution percentages, click for expert details

#### 3.2 ROC and Precision-Recall Curves
- **Chart Type**: Line charts (multiple)
- **Data Source**: Model evaluation metrics
- **Implementation**:
  ```javascript
  function createPerformanceCurves(rocData, prData) {
    // ROC Curve
    const rocLayout = {
      title: 'ROC Curve',
      xaxis: {title: 'False Positive Rate', range: [0, 1]},
      yaxis: {title: 'True Positive Rate', range: [0, 1]},
      shapes: [{
        type: 'line',
        x0: 0, y0: 0,
        x1: 1, y1: 1,
        line: {dash: 'dash', color: 'gray'}
      }]
    };
    
    const rocTrace = {
      x: rocData.fpr,
      y: rocData.tpr,
      mode: 'lines',
      type: 'scatter',
      name: `ROC (AUC = ${rocData.auc.toFixed(3)})`,
      line: {color: '#F44336'}
    };
    
    // Precision-Recall Curve
    const prLayout = {
      title: 'Precision-Recall Curve',
      xaxis: {title: 'Recall', range: [0, 1]},
      yaxis: {title: 'Precision', range: [0, 1]}
    };
    
    const prTrace = {
      x: prData.recall,
      y: prData.precision,
      mode: 'lines',
      type: 'scatter',
      name: `PR (AP = ${prData.average_precision.toFixed(3)})`,
      line: {color: '#4CAF50'}
    };
    
    Plotly.newPlot('roc-curve', [rocTrace], rocLayout);
    Plotly.newPlot('pr-curve', [prTrace], prLayout);
  }
  ```
- **Interactivity**: Toggle between different model runs, zoom to compare performance

#### 3.3 Confusion Matrix Visualization
- **Chart Type**: Interactive heatmap
- **Data Source**: Model evaluation results
- **Implementation**:
  ```javascript
  function createConfusionMatrix(confusionMatrix, labels) {
    const layout = {
      title: 'Confusion Matrix',
      xaxis: {title: 'Predicted', tickvals: [...Array(labels.length).keys()], ticktext: labels},
      yaxis: {title: 'Actual', tickvals: [...Array(labels.length).keys()], ticktext: labels},
      annotations: []
    };
    
    // Add text annotations to each cell
    for (let i = 0; i < confusionMatrix.length; i++) {
      for (let j = 0; j < confusionMatrix[i].length; j++) {
        const value = confusionMatrix[i][j];
        const total = confusionMatrix.flat().reduce((a, b) => a + b, 0);
        const percentage = (value / total * 100).toFixed(1);
        
        layout.annotations.push({
          x: j,
          y: i,
          text: `${value}<br>(${percentage}%)`,
          font: {color: 'white'},
          showarrow: false
        });
      }
    }
    
    const heatmapTrace = {
      z: confusionMatrix,
      x: labels,
      y: labels,
      type: 'heatmap',
      colorscale: [
        [0, '#F8F9FA'],
        [1, '#4A148C']
      ],
      showscale: false
    };
    
    Plotly.newPlot('confusion-matrix', [heatmapTrace], layout);
  }
  ```
- **Interactivity**: Hover for cell details, toggle between count and percentage views

### 4. Drift Analysis Visualizations

#### 4.1 Temporal Drift Monitoring
- **Chart Type**: Time series line chart with confidence intervals
- **Data Source**: Model performance over time
- **Implementation**:
  ```javascript
  function createDriftMonitoring(timePoints, performance, confidence) {
    const upper = performance.map((val, i) => val + confidence[i]);
    const lower = performance.map((val, i) => val - confidence[i]);
    
    const layout = {
      title: 'Performance Drift Over Time',
      xaxis: {title: 'Date'},
      yaxis: {title: 'Performance Metric', range: [0, 1]},
      shapes: [{
        type: 'rect',
        x0: timePoints[0],
        x1: timePoints[timePoints.length - 1],
        y0: 0.8,
        y1: 1,
        fillcolor: 'rgba(0, 255, 0, 0.1)',
        line: {width: 0}
      }, {
        type: 'rect',
        x0: timePoints[0],
        x1: timePoints[timePoints.length - 1],
        y0: 0.6,
        y1: 0.8,
        fillcolor: 'rgba(255, 255, 0, 0.1)',
        line: {width: 0}
      }, {
        type: 'rect',
        x0: timePoints[0],
        x1: timePoints[timePoints.length - 1],
        y0: 0,
        y1: 0.6,
        fillcolor: 'rgba(255, 0, 0, 0.1)',
        line: {width: 0}
      }]
    };
    
    const performanceTrace = {
      x: timePoints,
      y: performance,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Performance',
      line: {color: '#673AB7'}
    };
    
    const upperTrace = {
      x: timePoints,
      y: upper,
      type: 'scatter',
      mode: 'lines',
      name: 'Upper CI',
      line: {width: 0},
      showlegend: false
    };
    
    const lowerTrace = {
      x: timePoints,
      y: lower,
      type: 'scatter',
      mode: 'lines',
      name: 'Lower CI',
      line: {width: 0},
      fill: 'tonexty',
      fillcolor: 'rgba(103, 58, 183, 0.3)',
      showlegend: false
    };
    
    Plotly.newPlot('drift-monitoring', [upperTrace, performanceTrace, lowerTrace], layout);
  }
  ```
- **Interactivity**: Metric selector dropdown, time range slider, drift alert thresholds

#### 4.2 Feature Drift Visualization
- **Chart Type**: Parallel coordinates plot
- **Data Source**: Feature distributions over time
- **Implementation**:
  ```javascript
  function createFeatureDriftPlot(features, timePoints, distributions) {
    const dimensions = features.map(feature => ({
      label: feature,
      values: distributions[feature],
      range: [Math.min(...distributions[feature]) * 0.9, Math.max(...distributions[feature]) * 1.1]
    }));
    
    const layout = {
      title: 'Feature Distribution Drift',
      margin: {l: 80, r: 80, t: 100, b: 80}
    };
    
    const parallelTrace = {
      type: 'parcoords',
      line: {
        color: timePoints.map((date, i) => i),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: {title: 'Time Period'}
      },
      dimensions: dimensions
    };
    
    Plotly.newPlot('feature-drift', [parallelTrace], layout);
  }
  ```
- **Interactivity**: Feature selector, time period comparison, distribution type toggle

### 5. Clinical Decision Support Visualizations

#### 5.1 Patient Risk Timeline
- **Chart Type**: Timeline with risk heatmap
- **Data Source**: Predicted risk over time for individual patients
- **Implementation**:
  ```javascript
  function createRiskTimeline(patientId, timestamps, riskScores, events) {
    const layout = {
      title: `Migraine Risk Timeline for Patient #${patientId}`,
      xaxis: {title: 'Date', type: 'date'},
      yaxis: {title: 'Risk Score', range: [0, 1]},
      shapes: []
    };
    
    // Add shapes for actual migraine events
    events.forEach(event => {
      layout.shapes.push({
        type: 'line',
        x0: event,
        x1: event,
        y0: 0,
        y1: 1,
        line: {color: 'red', width: 2, dash: 'dash'}
      });
    });
    
    const riskTrace = {
      x: timestamps,
      y: riskScores,
      type: 'scatter',
      mode: 'lines',
      name: 'Risk Score',
      line: {color: '#FF5722'},
      fill: 'tozeroy',
      fillcolor: 'rgba(255, 87, 34, 0.3)'
    };
    
    const eventTrace = {
      x: events,
      y: events.map(() => 0.95),
      type: 'scatter',
      mode: 'markers',
      name: 'Migraine Event',
      marker: {symbol: 'star', size: 12, color: 'red'}
    };
    
    Plotly.newPlot('risk-timeline', [riskTrace, eventTrace], layout);
  }
  ```
- **Interactivity**: Patient selector, time range slider, threshold adjustment

#### 5.2 Trigger Pattern Analysis
- **Chart Type**: Radar chart
- **Data Source**: Feature importance for trigger categories
- **Implementation**:
  ```javascript
  function createTriggerRadarChart(triggerCategories, importanceScores, patientId) {
    const layout = {
      title: `Trigger Pattern Analysis for Patient #${patientId}`,
      polar: {
        radialaxis: {visible: true, range: [0, 1]}
      }
    };
    
    const radarTrace = {
      type: 'scatterpolar',
      r: [...importanceScores, importanceScores[0]], // Close the loop
      theta: [...triggerCategories, triggerCategories[0]], // Close the loop
      fill: 'toself',
      fillcolor: 'rgba(25, 118, 210, 0.4)',
      line: {color: '#1976D2'}
    };
    
    Plotly.newPlot('trigger-radar', [radarTrace], layout);
  }
  ```
- **Interactivity**: Patient comparison overlay, trigger category filtering

### 6. Advanced MoE-Specific Visualizations

#### 6.1 Expert Model Specialization Map
- **Chart Type**: Network diagram / force-directed graph
- **Data Source**: Expert model specialization areas and feature relationships
- **Implementation**:
  ```javascript
  function createExpertSpecializationMap(experts, features, relationships) {
    const nodes = [
      // Expert nodes
      ...experts.map(expert => ({
        id: expert.id,
        label: expert.name,
        group: 'expert',
        size: 30,
        color: '#E91E63'
      })),
      
      // Feature nodes
      ...features.map(feature => ({
        id: feature.id,
        label: feature.name,
        group: 'feature',
        size: 20,
        color: '#2196F3'
      }))
    ];
    
    const edges = relationships.map(rel => ({
      from: rel.expert_id,
      to: rel.feature_id,
      value: rel.importance,
      title: `Importance: ${rel.importance.toFixed(2)}`,
      width: Math.max(1, rel.importance * 10),
      color: {
        color: '#999',
        opacity: Math.max(0.2, rel.importance)
      }
    }));
    
    const container = document.getElementById('expert-specialization-map');
    const data = {nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges)};
    const options = {
      physics: {
        stabilization: true,
        barnesHut: {gravitationalConstant: -80000, springConstant: 0.001}
      },
      nodes: {shape: 'dot', font: {size: 14}},
      edges: {smooth: {type: 'continuous'}}
    };
    
    new vis.Network(container, data, options);
  }
  ```
- **Interactivity**: Zoom, pan, click nodes for details, highlight connections

#### 6.2 Gating Network Decision Boundaries
- **Chart Type**: 3D surface plot with contours
- **Data Source**: Gating network decision function outputs
- **Implementation**:
  ```javascript
  function createGatingDecisionBoundaries(featureX, featureY, expertsWeights) {
    // Create 3D surfaces for each expert's gating weight
    const data = expertsWeights.map((weights, expertIndex) => ({
      z: weights,
      x: featureX,
      y: featureY,
      type: 'surface',
      name: `Expert ${expertIndex+1}`,
      colorscale: expertColorscales[expertIndex],
      opacity: 0.7,
      showscale: true
    }));
    
    // Add contour plots for each expert
    expertsWeights.forEach((weights, expertIndex) => {
      data.push({
        z: weights,
        x: featureX,
        y: featureY,
        type: 'contour',
        name: `Expert ${expertIndex+1} Boundary`,
        showscale: false,
        contours: {
          coloring: 'lines',
          showlabels: true
        },
        line: {color: expertColors[expertIndex]}
      });
    });
    
    const layout = {
      title: 'Gating Network Decision Boundaries',
      scene: {
        xaxis: {title: featureX[0]},
        yaxis: {title: featureY[0]},
        zaxis: {title: 'Expert Weight'}
      },
      margin: {l: 65, r: 50, b: 65, t: 90}
    };
    
    Plotly.newPlot('gating-boundaries', data, layout);
  }
  ```
- **Interactivity**: 3D rotation, feature selector dropdowns, toggle between experts

#### 6.3 Ensemble Uncertainty Visualization
- **Chart Type**: Violin plots with confidence intervals
- **Data Source**: Expert model individual predictions with uncertainty estimates
- **Implementation**:
  ```javascript
  function createUncertaintyVisualization(patients, predictions, uncertainties, expertNames) {
    const data = expertNames.map((name, i) => ({
      type: 'violin',
      x: patients.map(() => name),
      y: predictions[i],
      box: {visible: true},
      meanline: {visible: true},
      points: 'all',
      pointpos: 0,
      jitter: 0.3,
      name: name,
      marker: {size: 8}
    }));
    
    // Add ensemble prediction with CI
    data.push({
      type: 'scatter',
      x: expertNames,
      y: uncertainties.map(u => u.mean),
      mode: 'markers',
      name: 'Ensemble Mean',
      marker: {size: 12, color: '#4CAF50'},
      error_y: {
        type: 'data',
        array: uncertainties.map(u => u.confidence),
        visible: true
      }
    });
    
    const layout = {
      title: 'Expert Model Agreement and Uncertainty',
      xaxis: {title: 'Expert Model'},
      yaxis: {title: 'Prediction Value', range: [0, 1]},
      violingap: 0.1,
      violingroupgap: 0.1
    };
    
    Plotly.newPlot('uncertainty-visualization', data, layout);
  }
  ```
- **Interactivity**: Patient selector, hover for statistics, toggle between violin/box/scatter

## Data Integration Approach

### 1. Clinical Data Integration Pipeline

```python
class ClinicalDataIntegrationPipeline:
    def __init__(self, config):
        self.validator = ClinicalDataValidator(config['validation_rules'])
        self.preprocessor = DataPreprocessor(config['preprocessing'])
        self.feature_engine = FeatureEngineer(config['feature_engineering'])
        self.cache = DataCache(config['cache_settings'])
    
    def process_clinical_data(self, data_path, patient_id=None):
        """Process clinical data through the pipeline"""
        # Load and validate
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Filter for specific patient if needed
        if patient_id is not None:
            data = data[data['patient_id'] == patient_id]
        
        # Validate
        validation_results = self.validator.validate(data)
        if not validation_results['is_valid']:
            return {
                'status': 'error',
                'validation_issues': validation_results['issues'],
                'data_quality_score': validation_results['quality_score']
            }
        
        # Preprocess
        processed_data = self.preprocessor.process(data)
        
        # Generate features
        features = self.feature_engine.generate_features(processed_data)
        
        # Cache processed data and features
        cache_key = f"{hashlib.md5(data_path.encode()).hexdigest()}_{patient_id or 'all'}"
        self.cache.store(cache_key, {
            'processed_data': processed_data,
            'features': features,
            'validation_results': validation_results
        })
        
        return {
            'status': 'success',
            'processed_data': processed_data,
            'features': features,
            'validation_results': validation_results,
            'cache_key': cache_key
        }
```

### 2. MoE Model Integration API

```python
class MoEModelAPI:
    def __init__(self, model_path=None):
        self.model = self._load_or_create_model(model_path)
        self.explainer = None
    
    def _load_or_create_model(self, model_path):
        """Load existing model or create new one"""
        if model_path and os.path.exists(model_path):
            return MoEModel.load(model_path)
        else:
            return MoEModel()
    
    def train(self, features, targets, expert_config=None):
        """Train the MoE model and initialize explainer"""
        training_results = self.model.train(features, targets, expert_config)
        self.explainer = self._create_explainer(features)
        return training_results
    
    def predict(self, features, explanation=True):
        """Generate predictions with optional explanation"""
        predictions = self.model.predict(features)
        expert_weights = self.model.get_expert_weights(features)
        
        result = {
            'predictions': predictions,
            'expert_weights': expert_weights
        }
        
        if explanation and self.explainer:
            try:
                shap_values = self.explainer.explain(self.model, features)
                result['explanations'] = {
                    'shap_values': shap_values,
                    'feature_importance': self._get_feature_importance(shap_values, features.columns)
                }
            except Exception as e:
                result['explanations'] = {
                    'error': f"Explanation generation failed: {str(e)}",
                    'feature_importance': self.model.get_feature_importance()
                }
        
        return result
    
    def _create_explainer(self, features):
        """Create SHAP explainer for the model"""
        try:
            from explainability.shap_explainer import create_explainer
            return create_explainer(self.model, features)
        except Exception as e:
            warnings.warn(f"Could not create explainer: {str(e)}")
            return None
    
    def _get_feature_importance(self, shap_values, feature_names):
        """Extract feature importance from SHAP values"""
        if hasattr(shap_values, 'values'):
            values = np.abs(shap_values.values).mean(axis=0)
        else:
            values = np.abs(shap_values).mean(axis=0)
        
        return [{'feature': feature, 'importance': float(importance)} 
                for feature, importance in zip(feature_names, values)]
```

## Interactive Dashboard Features

### 1. Real-time Data Processing Workflow

- **Drag-and-drop Data Upload**:
  ```javascript
  function setupDataUpload() {
    const dropzone = document.getElementById('data-dropzone');
    
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-active');
    });
    
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('drag-active');
    });
    
    dropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-active');
      
      const files = e.dataTransfer.files;
      if (files.length) {
        // Update UI to show processing
        showProcessingState();
        
        // Process each file
        for (const file of files) {
          try {
            // Upload and process file
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/data/upload', {
              method: 'POST',
              body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
              // Update data catalog
              updateDataCatalog(result.datasetInfo);
              
              // Show validation results
              displayValidationResults(result.validationResults);
              
              // Generate data quality visualizations
              createDataQualityCharts(result.qualityMetrics);
            } else {
              showError(`Processing failed: ${result.error}`);
            }
          } catch (error) {
            showError(`Upload failed: ${error.message}`);
          }
        }
        
        // Hide processing state
        hideProcessingState();
      }
    });
  }
  ```

### 2. AI-Assisted Data Exploration

- **Smart Feature Recommender**:
  ```javascript
  function setupFeatureRecommender() {
    const featureInput = document.getElementById('feature-search');
    const recommendationsContainer = document.getElementById('feature-recommendations');
    
    featureInput.addEventListener('input', async () => {
      const query = featureInput.value.trim();
      if (query.length < 2) {
        recommendationsContainer.innerHTML = '';
        return;
      }
      
      try {
        const response = await fetch(`/api/features/recommend?query=${encodeURIComponent(query)}`);
        const recommendations = await response.json();
        
        // Clear previous recommendations
        recommendationsContainer.innerHTML = '';
        
        // Display recommendations with relevance scores
        recommendations.forEach(rec => {
          const item = document.createElement('div');
          item.className = 'recommendation-item';
          item.innerHTML = `
            <span class="feature-name">${rec.feature}</span>
            <span class="relevance-score">${(rec.relevance * 100).toFixed(0)}%</span>
            <span class="feature-description">${rec.description}</span>
          `;
          
          // Add click handler to select this feature
          item.addEventListener('click', () => {
            selectFeature(rec.feature);
            recommendationsContainer.innerHTML = '';
          });
          
          recommendationsContainer.appendChild(item);
        });
      } catch (error) {
        console.error('Feature recommendation failed:', error);
      }
    });
  }
  ```

### 3. Interactive Patient Cohort Selection

- **Dynamic Cohort Builder**:
  ```javascript
  class CohortBuilder {
    constructor(container) {
      this.container = document.getElementById(container);
      this.criteria = [];
      this.patients = [];
      this.setupInterface();
    }
    
    setupInterface() {
      // Create cohort builder UI
      this.container.innerHTML = `
        <div class="cohort-builder">
          <h3>Build Patient Cohort</h3>
          <div class="criteria-container" id="criteria-list"></div>
          <button id="add-criterion" class="btn btn-primary">Add Criterion</button>
          <div class="selected-cohort">
            <div class="count-badge" id="patient-count">0 patients selected</div>
            <button id="apply-cohort" class="btn btn-success" disabled>Apply Cohort</button>
          </div>
        </div>
      `;
      
      // Add event listeners
      document.getElementById('add-criterion').addEventListener('click', () => this.addCriterion());
      document.getElementById('apply-cohort').addEventListener('click', () => this.applyCohort());
    }
    
    addCriterion() {
      const criterionId = `criterion-${Date.now()}`;
      const criterionElement = document.createElement('div');
      criterionElement.className = 'criterion';
      criterionElement.id = criterionId;
      
      criterionElement.innerHTML = `
        <select class="feature-select">
          <option value="">Select Feature</option>
          ${this.getFeatureOptions()}
        </select>
        <select class="operator-select">
          <option value="eq">equals</option>
          <option value="gt">greater than</option>
          <option value="lt">less than</option>
          <option value="between">between</option>
          <option value="in">in list</option>
        </select>
        <div class="value-container">
          <input type="text" class="value-input" placeholder="Value">
        </div>
        <button class="remove-btn"><i class="fa fa-times"></i></button>
      `;
      
      document.getElementById('criteria-list').appendChild(criterionElement);
      
      // Add event listeners for this criterion
      const removeBtn = criterionElement.querySelector('.remove-btn');
      removeBtn.addEventListener('click', () => this.removeCriterion(criterionId));
      
      const featureSelect = criterionElement.querySelector('.feature-select');
      featureSelect.addEventListener('change', () => this.updateOperators(criterionId));
      
      const operatorSelect = criterionElement.querySelector('.operator-select');
      operatorSelect.addEventListener('change', () => this.updateValueInput(criterionId));
      
      // Update criterion list and preview cohort
      this.criteria.push({
        id: criterionId,
        feature: '',
        operator: 'eq',
        value: ''
      });
      
      // Attach input event listeners for value changes
      const valueInput = criterionElement.querySelector('.value-input');
      valueInput.addEventListener('input', () => this.updateCriterion(criterionId));
      
      this.updateCohortPreview();
    }
    
    removeCriterion(id) {
      // Remove from DOM
      const element = document.getElementById(id);
      if (element) element.remove();
      
      // Remove from criteria list
      this.criteria = this.criteria.filter(c => c.id !== id);
      
      // Update preview
      this.updateCohortPreview();
    }
    
    updateOperators(id) {
      const criterion = this.criteria.find(c => c.id === id);
      const element = document.getElementById(id);
      const featureSelect = element.querySelector('.feature-select');
      
      criterion.feature = featureSelect.value;
      this.updateValueInput(id);
      this.updateCohortPreview();
    }
    
    updateValueInput(id) {
      const criterion = this.criteria.find(c => c.id === id);
      const element = document.getElementById(id);
      const operatorSelect = element.querySelector('.operator-select');
      const valueContainer = element.querySelector('.value-container');
      
      criterion.operator = operatorSelect.value;
      
      // Update value input based on operator
      if (operatorSelect.value === 'between') {
        valueContainer.innerHTML = `
          <input type="text" class="value-input-min" placeholder="Min">
          <span class="and-separator">and</span>
          <input type="text" class="value-input-max" placeholder="Max">
        `;
        
        // Add event listeners
        const minInput = valueContainer.querySelector('.value-input-min');
        const maxInput = valueContainer.querySelector('.value-input-max');
        
        minInput.addEventListener('input', () => this.updateCriterion(id));
        maxInput.addEventListener('input', () => this.updateCriterion(id));
      } else if (operatorSelect.value === 'in') {
        valueContainer.innerHTML = `
          <textarea class="value-input-list" placeholder="Comma-separated values"></textarea>
        `;
        
        // Add event listener
        const listInput = valueContainer.querySelector('.value-input-list');
        listInput.addEventListener('input', () => this.updateCriterion(id));
      } else {
        valueContainer.innerHTML = `
          <input type="text" class="value-input" placeholder="Value">
        `;
        
        // Add event listener
        const valueInput = valueContainer.querySelector('.value-input');
        valueInput.addEventListener('input', () => this.updateCriterion(id));
      }
      
      this.updateCriterion(id);
    }
    
    updateCriterion(id) {
      const criterion = this.criteria.find(c => c.id === id);
      const element = document.getElementById(id);
      
      if (criterion.operator === 'between') {
        const minInput = element.querySelector('.value-input-min');
        const maxInput = element.querySelector('.value-input-max');
        criterion.value = {min: minInput.value, max: maxInput.value};
      } else if (criterion.operator === 'in') {
        const listInput = element.querySelector('.value-input-list');
        criterion.value = listInput.value.split(',').map(v => v.trim());
      } else {
        const valueInput = element.querySelector('.value-input');
        criterion.value = valueInput.value;
      }
      
      this.updateCohortPreview();
    }
    
    async updateCohortPreview() {
      // Only preview if we have valid criteria
      if (!this.hasValidCriteria()) {
        document.getElementById('patient-count').textContent = '0 patients selected';
        document.getElementById('apply-cohort').disabled = true;
        return;
      }
      
      try {
        const response = await fetch('/api/patients/filter', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({criteria: this.criteria})
        });
        
        const result = await response.json();
        this.patients = result.patients;
        
        // Update count display
        const countElement = document.getElementById('patient-count');
        countElement.textContent = `${this.patients.length} patients selected`;
        
        // Enable/disable apply button
        document.getElementById('apply-cohort').disabled = this.patients.length === 0;
      } catch (error) {
        console.error('Failed to preview cohort:', error);
      }
    }
    
    hasValidCriteria() {
      if (this.criteria.length === 0) return false;
      
      return this.criteria.every(criterion => {
        if (!criterion.feature) return false;
        
        if (criterion.operator === 'between') {
          return criterion.value.min !== '' && criterion.value.max !== '';
        } else if (criterion.operator === 'in') {
          return criterion.value.length > 0 && criterion.value[0] !== '';
        } else {
          return criterion.value !== '';
        }
      });
    }
    
    applyCohort() {
      // Trigger event with selected patients
      const event = new CustomEvent('cohortselected', {
        detail: {
          criteria: this.criteria,
          patients: this.patients
        }
      });
      
      document.dispatchEvent(event);
    }
    
    getFeatureOptions() {
      // This would be populated dynamically from available features
      return `
        <option value="age">Age</option>
        <option value="gender">Gender</option>
        <option value="migraine_frequency">Migraine Frequency</option>
        <option value="medication_response">Medication Response</option>
        <option value="sleep_quality">Sleep Quality</option>
      `;
    }
  }
  ```

## Expected Deliverables

Please provide:

1. Complete source code for the dashboard application
2. Docker configuration for containerized deployment
3. Installation and usage documentation
4. Screenshots or demo of the functioning dashboard

## Additional Considerations

- The dashboard should be intuitive for clinical users
- Visualizations should provide actionable insights
- Error handling should be robust and user-friendly
- The system should be designed for future extensibility

## Example User Flows

### User Flow 1: Clinical Data Analysis
1. User uploads clinical data CSV file
2. System validates data structure and quality
3. Dashboard displays data quality metrics and distributions
4. User reviews validation results and proceeds to modeling

### User Flow 2: Migraine Prediction
1. User selects preprocessed data for analysis
2. System configures and trains MoE model
3. Dashboard displays prediction results with confidence scores
4. User explores SHAP feature importance visualizations
5. System generates downloadable report with findings

### User Flow 3: Model Performance Monitoring
1. User selects historical model runs for comparison
2. Dashboard displays performance metrics across runs
3. System highlights drift patterns and data quality changes
4. User explores feature importance stability over time

## Technical Implementation Details

The dashboard should implement these key technical components:

```python
# Example: Data processing integration
def process_clinical_data(uploaded_file):
    """Process uploaded clinical data using the validator pipeline"""
    # Load data from uploaded file
    clinical_data = pd.read_csv(uploaded_file)
    
    # Import the validator
    from data_integration.clinical_data_validator import ClinicalDataValidator
    
    # Create validator instance and run validation
    validator = ClinicalDataValidator()
    validation_results = validator.validate(clinical_data)
    
    # Return processed data and validation results
    return clinical_data, validation_results

# Example: MoE model integration
def run_moe_prediction(clinical_data):
    """Run MoE model prediction with SHAP analysis"""
    # Preprocess data for model
    features = preprocess_for_model(clinical_data)
    
    # Import MoE model
    from moe.moe_model import MoEModel
    
    # Load or create model
    model = MoEModel.load_or_create()
    
    # Generate predictions
    predictions = model.predict(features)
    
    # Run SHAP analysis
    from explainability.shap_explainer import explain_model
    shap_values = explain_model(model, features)
    
    return predictions, shap_values
```

## Dashboard Mockup Guidance

The dashboard should follow this general layout:

```
+---------------------------------------------------------------+
|                      MIGRAINE MoE DASHBOARD                   |
+----------------+----------------------------------------------+
| NAVIGATION     |                                              |
|                |                                              |
| • Overview     |          MAIN CONTENT AREA                   |
| • Data         |          (Changes based on selection)        |
| • Model        |                                              |
| • Predictions  |                                              |
| • Performance  |                                              |
| • Settings     |                                              |
|                |                                              |
+----------------+----------------------------------------------+
|                       STATUS & NOTIFICATIONS                  |
+---------------------------------------------------------------+
```

Please create an intuitive, professional dashboard that enables clinical users to effectively leverage the MoE framework for migraine prediction and analysis.
