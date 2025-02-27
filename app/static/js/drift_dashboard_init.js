/**
 * Drift Dashboard Initialization Script
 * 
 * This script initializes the drift detection dashboard when the page loads.
 * It dynamically creates all necessary elements since we can't modify the HTML template directly.
 */
document.addEventListener('DOMContentLoaded', function() {
    // Load CSS
    const cssLink = document.createElement('link');
    cssLink.rel = 'stylesheet';
    cssLink.href = '/static/css/drift_dashboard.css';
    document.head.appendChild(cssLink);
    
    // Get the main content area
    const mainContent = document.querySelector('main') || document.body;
    
    // Clear any existing content if needed
    // mainContent.innerHTML = '';
    
    // Create the complete dashboard structure
    const dashboardHTML = `
        <div id="driftDashboardContainer" class="drift-dashboard">
            <div class="drift-header">
                <h2>Drift Detection Dashboard</h2>
                <div class="drift-controls">
                    <div class="control-group">
                        <label for="timeRange">Time Range:</label>
                        <select id="timeRange" name="timeRange">
                            <option value="7">Last 7 days</option>
                            <option value="30" selected>Last 30 days</option>
                            <option value="90">Last 90 days</option>
                            <option value="180">Last 180 days</option>
                        </select>
                    </div>
                    <button id="refreshDrift" class="btn btn-primary">Refresh Data</button>
                </div>
            </div>
            
            <div id="driftStatusContainer" class="drift-status">
                <div id="driftLoading" class="loading-indicator">
                    <span>Loading drift data...</span>
                </div>
                <div id="driftError" class="error-message"></div>
                <div id="driftNoData" class="no-data-message">
                    <p>No drift data available for the selected time period.</p>
                </div>
            </div>
            
            <div class="chart-row">
                <div id="severityChartContainer" class="chart-container">
                    <h3>Drift Severity Over Time</h3>
                    <canvas id="severityChart"></canvas>
                </div>
            </div>
            
            <div class="chart-row">
                <div id="featureDriftContainer" class="chart-container">
                    <h3>Feature Drift Frequency</h3>
                    <canvas id="featureDriftChart"></canvas>
                </div>
            </div>
            
            <div id="driftSummary" class="drift-summary">
                <h3>Drift Detection Summary</h3>
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-value" id="totalDrifts">0</span>
                        <span class="stat-label">Total Drifts</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" id="maxSeverity">0.000</span>
                        <span class="stat-label">Max Severity</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" id="avgSeverity">0.000</span>
                        <span class="stat-label">Avg Severity</span>
                    </div>
                </div>
                <div class="top-features">
                    <h4>Most Affected Features:</h4>
                    <ul id="topFeaturesList">
                        <li>No drift detected yet</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    // Insert the dashboard HTML
    mainContent.innerHTML += dashboardHTML;
    
    // Load Chart.js if not already loaded
    if (typeof Chart === 'undefined') {
        const chartScript = document.createElement('script');
        chartScript.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        document.head.appendChild(chartScript);
        
        // Load Chart.js annotation plugin
        const annotationScript = document.createElement('script');
        annotationScript.src = 'https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation';
        document.head.appendChild(annotationScript);
        
        // Initialize dashboard after Chart.js loads
        chartScript.onload = function() {
            annotationScript.onload = function() {
                initializeDashboard();
            };
        };
    } else {
        // Chart.js already loaded, initialize dashboard
        initializeDashboard();
    }
    
    // Add event listeners
    document.getElementById('timeRange').addEventListener('change', function() {
        loadDriftData(parseInt(this.value));
    });
    
    document.getElementById('refreshDrift').addEventListener('click', function() {
        loadDriftData(parseInt(document.getElementById('timeRange').value));
    });
    
    function initializeDashboard() {
        // Initialize charts
        initializeCharts();
        
        // Load initial data
        loadDriftData(30);
    }
    
    function initializeCharts() {
        // Initialize severity chart
        const severityCtx = document.getElementById('severityChart').getContext('2d');
        window.severityChart = new Chart(severityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Drift Severity',
                    data: [],
                    borderColor: '#FF5722',
                    backgroundColor: 'rgba(255, 87, 34, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Drift Severity Over Time'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Severity: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            thresholdLine: {
                                type: 'line',
                                yMin: 0.6,
                                yMax: 0.6,
                                borderColor: 'rgba(255, 0, 0, 0.5)',
                                borderWidth: 1,
                                borderDash: [5, 5],
                                label: {
                                    content: 'Threshold',
                                    enabled: true,
                                    position: 'end'
                                }
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Severity'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });

        // Initialize feature drift chart
        const featureCtx = document.getElementById('featureDriftChart').getContext('2d');
        window.featureChart = new Chart(featureCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Drift Frequency',
                    data: [],
                    backgroundColor: '#2196F3'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Drift Frequency'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Feature'
                        }
                    }
                }
            }
        });
    }
    
    async function loadDriftData(days = 30) {
        try {
            // Show loading indicator
            showLoading(true);
            
            // Fetch data from API
            const response = await fetch(`/api/drift/visualization?days=${days}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update charts with new data
            updateCharts(data);
            
            // Hide loading indicator
            showLoading(false);
            
        } catch (error) {
            console.error('Error loading drift data:', error);
            showError('Failed to load drift data. Please try again later.');
            showLoading(false);
        }
    }
    
    function updateCharts(data) {
        if (data.timestamps && data.timestamps.length > 0) {
            // Format timestamps
            const labels = data.timestamps.map(ts => {
                const date = new Date(ts * 1000);
                return date.toLocaleDateString();
            });
            
            // Update severity chart
            window.severityChart.data.labels = labels;
            window.severityChart.data.datasets[0].data = data.severities;
            
            // Add drift point annotations
            const annotations = {};
            
            // Clear previous annotations except threshold
            Object.keys(window.severityChart.options.plugins.annotation.annotations).forEach(key => {
                if (key !== 'thresholdLine') {
                    delete window.severityChart.options.plugins.annotation.annotations[key];
                }
            });
            
            // Add drift point annotations
            data.drift_points.forEach((point, index) => {
                if (point < labels.length) {
                    annotations[`drift${index}`] = {
                        type: 'line',
                        xMin: point,
                        xMax: point,
                        borderColor: 'rgba(255, 0, 0, 0.7)',
                        borderWidth: 2
                    };
                }
            });
            
            // Update annotations
            window.severityChart.options.plugins.annotation.annotations = {
                ...window.severityChart.options.plugins.annotation.annotations,
                ...annotations
            };
            
            window.severityChart.update();
            
            // Update feature drift chart
            if (data.feature_drifts) {
                const features = Object.keys(data.feature_drifts);
                const driftCounts = features.map(f => data.feature_drifts[f]);
                
                window.featureChart.data.labels = features;
                window.featureChart.data.datasets[0].data = driftCounts;
                window.featureChart.update();
            }
            
            // Show summary
            updateSummary(data);
            
            // Hide no data message
            document.getElementById('driftNoData').style.display = 'none';
        } else {
            // Show no data message
            document.getElementById('driftNoData').style.display = 'block';
            
            // Clear charts
            window.severityChart.data.labels = [];
            window.severityChart.data.datasets[0].data = [];
            window.severityChart.update();
            
            window.featureChart.data.labels = [];
            window.featureChart.data.datasets[0].data = [];
            window.featureChart.update();
        }
    }
    
    function updateSummary(data) {
        // Calculate summary statistics
        const totalDrifts = data.drift_points.length;
        const maxSeverity = data.severities.length > 0 ? Math.max(...data.severities) : 0;
        const avgSeverity = data.severities.length > 0 ? 
            data.severities.reduce((sum, val) => sum + val, 0) / data.severities.length : 0;
        
        // Update summary values
        document.getElementById('totalDrifts').textContent = totalDrifts;
        document.getElementById('maxSeverity').textContent = maxSeverity.toFixed(3);
        document.getElementById('avgSeverity').textContent = avgSeverity.toFixed(3);
        
        // Most affected features
        const featureDrifts = Object.entries(data.feature_drifts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
        
        // Update feature list
        const featureList = document.getElementById('topFeaturesList');
        
        if (featureDrifts.length > 0) {
            featureList.innerHTML = featureDrifts.map(([feature, count]) => 
                `<li>${feature}: ${count} drifts</li>`).join('');
        } else {
            featureList.innerHTML = '<li>No drift detected yet</li>';
        }
    }
    
    function showLoading(isLoading) {
        const loadingEl = document.getElementById('driftLoading');
        loadingEl.style.display = isLoading ? 'flex' : 'none';
    }
    
    function showError(message) {
        const errorEl = document.getElementById('driftError');
        errorEl.textContent = message;
        errorEl.style.display = 'block';
        
        // Hide after 5 seconds
        setTimeout(() => {
            errorEl.style.display = 'none';
        }, 5000);
    }
});
