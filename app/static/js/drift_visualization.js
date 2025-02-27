/**
 * Drift Visualization Component
 * 
 * Provides interactive visualizations for concept drift detection and analysis
 */
class DriftVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.severityChart = null;
        this.featureChart = null;
        this._createChartContainers();
        this._createControls();
        this.initializeCharts();
        this.loadData();
    }

    async loadData(days = 30) {
        this._showLoading(true);
        try {
            // Load drift detection data
            const driftResponse = await fetch(`/api/dashboard/drift-data?days=${days}`);
            if (!driftResponse.ok) throw new Error('Failed to load drift data');
            const driftData = await driftResponse.json();

            // Update visualization
            this.updateDriftCharts(driftData);
        } catch (error) {
            console.error('Error loading data:', error);
            this._showError(error.message);
        } finally {
            this._showLoading(false);
        }
    }

    _createChartContainers() {
        // Create containers for charts
        const driftContainer = document.createElement('div');
        driftContainer.className = 'chart-container';
        driftContainer.innerHTML = `
            <canvas id="severityChart"></canvas>
            <canvas id="featureChart"></canvas>
        `;
        this.container.appendChild(driftContainer);

        // Create controls
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'controls-container mb-4';
        controlsContainer.innerHTML = `
            <div class="row align-items-center">
                <div class="col-auto">
                    <select id="timeRange" class="form-select">
                        <option value="7">Last 7 days</option>
                        <option value="30" selected>Last 30 days</option>
                        <option value="90">Last 90 days</option>
                        <option value="180">Last 180 days</option>
                    </select>
                </div>
                <div class="col-auto">
                    <button id="refreshBtn" class="btn btn-primary">Refresh Data</button>
                </div>
            </div>
        `;
        this.container.insertBefore(controlsContainer, driftContainer);

        // Add event listeners
        document.getElementById('timeRange').addEventListener('change', (e) => {
            this.loadData(parseInt(e.target.value));
        });

        document.getElementById('refreshBtn').addEventListener('click', () => {
            const days = parseInt(document.getElementById('timeRange').value);
            this.loadData(days);
        });
    }

    _showLoading(show) {
        let loadingIndicator = document.getElementById('loadingIndicator');
        if (!loadingIndicator) {
            loadingIndicator = document.createElement('div');
            loadingIndicator.id = 'loadingIndicator';
            loadingIndicator.className = 'loading-indicator';
            loadingIndicator.innerHTML = `
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            `;
            document.body.appendChild(loadingIndicator);
        }
        loadingIndicator.style.display = show ? 'flex' : 'none';
    }

    _showError(message) {
        let errorDiv = document.getElementById('errorMessage');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'errorMessage';
            errorDiv.className = 'alert alert-danger';
            this.container.appendChild(errorDiv);
        }
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }

    initializeCharts() {
        // Initialize severity chart
        const severityCtx = document.getElementById('severityChart').getContext('2d');
        this.severityChart = new Chart(severityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Drift Severity',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Drift Severity Over Time'
                    },
                    annotation: {
                        annotations: {}
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Severity Score'
                        }
                    }
                }
            }
        });

        // Initialize feature drift chart
        const featureCtx = document.getElementById('featureChart').getContext('2d');
        this.featureChart = new Chart(featureCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Feature Drift Count',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature-wise Drift Distribution'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Drifts'
                        }
                    }
                }
            }
        });
    }

    updateDriftCharts(data) {
        if (!data.timestamps || !data.severities) {
            console.warn('No drift data available');
            this._showError('No drift data available');
            return;
        }

        // Format timestamps
        const labels = data.timestamps.map(ts => {
            const date = new Date(ts * 1000);
            return date.toLocaleDateString();
        });

        // Update severity chart
        this.severityChart.data.labels = labels;
        this.severityChart.data.datasets[0].data = data.severities;

        // Add drift threshold line
        const annotations = {
            thresholdLine: {
                type: 'line',
                yMin: data.drift_threshold,
                yMax: data.drift_threshold,
                borderColor: 'rgba(255, 0, 0, 0.5)',
                borderWidth: 2,
                borderDash: [6, 6],
                label: {
                    content: 'Drift Threshold',
                    enabled: true
                }
            }
        };

        // Add vertical lines for drift points
        if (data.drift_points) {
            data.drift_points.forEach((point, index) => {
                annotations[`drift${index}`] = {
                    type: 'line',
                    xMin: point,
                    xMax: point,
                    borderColor: 'rgba(255, 0, 0, 0.7)',
                    borderWidth: 2,
                    label: {
                        content: 'Drift Detected',
                        enabled: true
                    }
                };
            });
        }

        this.severityChart.options.plugins.annotation.annotations = annotations;
        this.severityChart.update();

        // Update feature drift chart
        if (data.feature_drifts) {
            const features = Object.keys(data.feature_drifts);
            const driftCounts = features.map(f => data.feature_drifts[f]);

            this.featureChart.data.labels = features;
            this.featureChart.data.datasets[0].data = driftCounts;
            this.featureChart.update();
        }
    }

    updatePerformanceCharts(data) {
        // Implementation for performance charts
        console.log('Performance data:', data);
    }

    updateOptimizationCharts(data) {
        // Implementation for optimization charts
        console.log('Optimization data:', data);
    }

    updateMetaAnalysisCharts(data) {
        // Implementation for meta analysis charts
        console.log('Meta analysis data:', data);
    }
}

// Add CSS to document head
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = `
        .drift-dashboard {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .drift-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .drift-header h2 {
            margin: 0;
            color: #333;
        }
        
        .drift-controls select {
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        
        .drift-status {
            display: flex;
            justify-content: space-between;
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #4CAF50;
            margin-right: 8px;
        }
        
        .status-dot.active {
            background-color: #F44336;
        }
        
        .drift-metrics {
            display: flex;
        }
        
        .metric {
            text-align: center;
            margin-left: 30px;
        }
        
        .metric-value {
            display: block;
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        
        .metric-label {
            font-size: 12px;
            color: #666;
        }
        
        .chart-row {
            display: flex;
            margin-bottom: 20px;
            gap: 20px;
        }
        
        .chart-container {
            flex: 1;
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .drift-details {
            flex: 1;
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            max-height: 300px;
            overflow-y: auto;
        }
        
        .drift-details h3 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #333;
        }
        
        .drift-event {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .event-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .event-timestamp {
            font-size: 12px;
            color: #666;
        }
        
        .event-severity {
            font-size: 12px;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 10px;
        }
        
        .severity-low {
            background-color: #E8F5E9;
            color: #2E7D32;
        }
        
        .severity-medium {
            background-color: #FFF8E1;
            color: #F57F17;
        }
        
        .severity-high {
            background-color: #FFEBEE;
            color: #C62828;
        }
        
        .event-details {
            display: flex;
            justify-content: space-between;
        }
        
        .event-feature {
            font-weight: 500;
        }
        
        .event-type {
            font-style: italic;
            color: #666;
        }
        
        .empty-state {
            color: #999;
            text-align: center;
            padding: 20px;
            font-style: italic;
        }
        
        .drift-summary {
            padding: 20px;
            background-color: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .summary-stats {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        .stat-item {
            text-align: center;
            margin-left: 30px;
        }
        
        .stat-value {
            display: block;
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
        }
        
        .top-features {
            margin-bottom: 20px;
        }
        
        .top-features ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .top-features li {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .top-features li:last-child {
            border-bottom: none;
        }
        
        .loading-indicator {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: none;
            justify-content: center;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        
        .error-message {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: none;
            justify-content: center;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            color: #F44336;
        }
        
        .no-data-message {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: none;
            justify-content: center;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            color: #999;
        }
    `;
    document.head.appendChild(style);
});
