/**
 * benchmark_dashboard.js
 * ---------------------
 * Handles visualization and interaction for the benchmarks tab
 */

class BenchmarkDashboard {
    constructor(parentDashboard) {
        if (!parentDashboard) {
            throw new Error('Parent dashboard is required');
        }
        
        this.parent = parentDashboard;
        this.charts = {};
        this.data = {
            benchmarkComparison: null,
            convergence: null,
            realDataPerformance: null,
            featureImportance: null
        };
        
        // Verify Chart.js is available
        if (typeof Chart === 'undefined') {
            throw new Error('Chart.js must be loaded before initializing BenchmarkDashboard');
        }
        
        // Initialize after dependencies are verified
        this.initialize().catch(error => {
            console.error('BenchmarkDashboard initialization failed:', error);
            this.parent.showError('Failed to initialize benchmark dashboard: ' + error.message);
        });
    }
    
    async initialize() {
        console.log('BenchmarkDashboard: Initializing...');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial data
        await this.loadAllData();
        
        console.log('BenchmarkDashboard: Initialization complete');
    }
    
    setupEventListeners() {
        const runBenchmarkButton = document.getElementById('runBenchmarkComparison');
        if (runBenchmarkButton) {
            runBenchmarkButton.addEventListener('click', () => this.runBenchmarkComparison());
        }
        
        const convergenceSelect = document.getElementById('convergenceFunction');
        if (convergenceSelect) {
            convergenceSelect.addEventListener('change', () => this.updateConvergencePlot());
        }
        
        const datasetSelect = document.getElementById('datasetSelector');
        if (datasetSelect) {
            datasetSelect.addEventListener('change', () => this.updateRealDataChart());
        }
    }
    
    async loadAllData() {
        try {
            this.parent.showLoading(true);
            
            const endpoints = [
                '/api/dashboard/benchmarks/comparison',
                '/api/dashboard/benchmarks/convergence',
                '/api/dashboard/benchmarks/real-data-performance',
                '/api/dashboard/benchmarks/feature-importance'
            ];
            
            const results = await Promise.allSettled(
                endpoints.map(endpoint => this.parent.fetchData(endpoint))
            );
            
            // Process results and handle errors individually
            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    const dataKeys = ['benchmarkComparison', 'convergence', 'realDataPerformance', 'featureImportance'];
                    this.data[dataKeys[index]] = result.value;
                } else {
                    console.error(`Failed to load ${endpoints[index]}:`, result.reason);
                }
            });
            
            // Initialize charts if we have any data
            if (Object.values(this.data).some(value => value !== null)) {
                this.initializeCharts();
            } else {
                throw new Error('No benchmark data could be loaded');
            }
        } catch (error) {
            console.error('Failed to load benchmark data:', error);
            throw error;
        } finally {
            this.parent.showLoading(false);
        }
    }
    
    initializeCharts() {
        this.initBenchmarkComparisonChart();
        this.initConvergenceChart();
        this.initRealDataChart();
        this.initFeatureImportanceChart();
    }
    
    /**
     * Initialize benchmark comparison chart
     */
    initBenchmarkComparisonChart() {
        const ctx = document.getElementById('benchmarkComparisonChart');
        if (!ctx || !this.data.benchmarkComparison) return;
        
        // Ensure any existing chart is destroyed first
        try {
            if (window.benchmarkComparisonChart && typeof window.benchmarkComparisonChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing global benchmark comparison chart');
                window.benchmarkComparisonChart.destroy();
            } else if (window.benchmarkComparisonChart) {
                console.warn('BenchmarkDashboard: Global benchmark comparison chart exists but destroy method is not available');
                window.benchmarkComparisonChart = null;
            }
            
            if (this.charts.benchmarkComparisonChart && typeof this.charts.benchmarkComparisonChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing instance benchmark comparison chart');
                this.charts.benchmarkComparisonChart.destroy();
            } else if (this.charts.benchmarkComparisonChart) {
                console.warn('BenchmarkDashboard: Instance benchmark comparison chart exists but destroy method is not available');
                this.charts.benchmarkComparisonChart = null;
            }
        } catch (error) {
            console.error('BenchmarkDashboard: Error destroying existing benchmark comparison chart:', error);
            // Reset chart references to avoid further issues
            window.benchmarkComparisonChart = null;
            this.charts.benchmarkComparisonChart = null;
        }
        
        // Extract data
        const algorithms = Object.keys(this.data.benchmarkComparison);
        const metrics = Object.keys(this.data.benchmarkComparison[algorithms[0]]).filter(m => m !== 'time');
        
        // Create datasets
        const datasets = metrics.map(metric => {
            return {
                label: metric.charAt(0).toUpperCase() + metric.slice(1).replace('_', ' '),
                data: algorithms.map(algo => this.data.benchmarkComparison[algo][metric]),
                backgroundColor: this.getMetricColor(metric, 0.7)
            };
        });
        
        // Create chart
        try {
            console.log('BenchmarkDashboard: Creating new benchmark comparison chart');
            this.charts.benchmarkComparisonChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: algorithms.map(a => a.toUpperCase()),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Score'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Performance Across Benchmark Functions'
                        }
                    }
                }
            });
            
            // Store reference globally to ensure we can destroy it later
            window.benchmarkComparisonChart = this.charts.benchmarkComparisonChart;
            console.log('BenchmarkDashboard: Successfully created benchmark comparison chart');
        } catch (error) {
            console.error('BenchmarkDashboard: Error creating benchmark comparison chart:', error);
        }
    }
    
    /**
     * Initialize convergence chart
     */
    initConvergenceChart(selectedFunction) {
        console.log('BenchmarkDashboard: Initializing convergence chart');
        const ctx = document.getElementById('convergenceChart');
        if (!ctx || !this.data.convergence) {
            console.warn('BenchmarkDashboard: Cannot initialize convergence chart - canvas or data missing');
            return;
        }
        
        // Ensure any existing chart is destroyed first
        try {
            if (window.convergenceChart && typeof window.convergenceChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing global convergence chart');
                window.convergenceChart.destroy();
            } else if (window.convergenceChart) {
                console.warn('BenchmarkDashboard: Global convergence chart exists but destroy method is not available');
                window.convergenceChart = null;
            }
            
            if (this.charts.convergenceChart && typeof this.charts.convergenceChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing instance convergence chart');
                this.charts.convergenceChart.destroy();
            } else if (this.charts.convergenceChart) {
                console.warn('BenchmarkDashboard: Instance convergence chart exists but destroy method is not available');
                this.charts.convergenceChart = null;
            }
        } catch (error) {
            console.error('BenchmarkDashboard: Error destroying existing convergence chart:', error);
            // Reset chart references to avoid further issues
            window.convergenceChart = null;
            this.charts.convergenceChart = null;
        }
        
        // Get selected function or use the select element
        if (!selectedFunction) {
            const select = document.getElementById('convergenceFunction');
            selectedFunction = select ? select.value : 'rosenbrock';
        }
        
        try {
            // Get data for optimizers
            const optimizers = Object.keys(this.data.convergence);
            
            // Create datasets
            const datasets = optimizers.map(optimizer => {
                return {
                    label: optimizer,
                    data: this.data.convergence[optimizer].performance,
                    borderColor: this.getOptimizerColor(optimizer),
                    backgroundColor: this.getOptimizerColor(optimizer, 0.1),
                    fill: false,
                    tension: 0.3
                };
            });
            
            // Create chart
            console.log('BenchmarkDashboard: Creating new convergence chart');
            this.charts.convergenceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: this.data.convergence[optimizers[0]].iterations,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 0.5, // Start from 0.5 to better show differences
                            max: 1.0,
                            title: {
                                display: true,
                                text: 'Performance Score'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Iterations'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `Convergence on ${selectedFunction} Function`
                        }
                    }
                }
            });
            
            // Store reference globally to ensure we can destroy it later
            window.convergenceChart = this.charts.convergenceChart;
            console.log('BenchmarkDashboard: Successfully created convergence chart');
        } catch (error) {
            console.error('BenchmarkDashboard: Error creating convergence chart:', error);
        }
    }
    
    /**
     * Initialize real data performance chart
     */
    initRealDataChart() {
        console.log('BenchmarkDashboard: Initializing real data performance chart');
        const ctx = document.getElementById('realDataChart');
        if (!ctx || !this.data.realDataPerformance) {
            console.warn('BenchmarkDashboard: Cannot initialize real data chart - canvas or data missing');
            return;
        }
        
        // Ensure any existing chart is destroyed first
        try {
            if (window.realDataChart && typeof window.realDataChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing global real data chart');
                window.realDataChart.destroy();
            } else if (window.realDataChart) {
                console.warn('BenchmarkDashboard: Global real data chart exists but destroy method is not available');
                window.realDataChart = null;
            }
            
            if (this.charts.realDataChart && typeof this.charts.realDataChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing instance real data chart');
                this.charts.realDataChart.destroy();
            } else if (this.charts.realDataChart) {
                console.warn('BenchmarkDashboard: Instance real data chart exists but destroy method is not available');
                this.charts.realDataChart = null;
            }
        } catch (error) {
            console.error('BenchmarkDashboard: Error destroying existing real data chart:', error);
            // Reset chart references to avoid further issues
            window.realDataChart = null;
            this.charts.realDataChart = null;
        }
        
        try {
            // Get metrics and algorithms
            const metrics = Object.keys(this.data.realDataPerformance);
            const algorithms = Object.keys(this.data.realDataPerformance[metrics[0]]);
            
            // Prepare datasets
            const datasets = algorithms.map(algo => {
                return {
                    label: algo,
                    data: metrics.map(metric => this.data.realDataPerformance[metric][algo]),
                    backgroundColor: this.getOptimizerColor(algo, 0.7),
                    borderWidth: 1
                };
            });
            
            // Create chart
            console.log('BenchmarkDashboard: Creating new real data performance chart');
            this.charts.realDataChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: metrics.map(m => m.toUpperCase()),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    elements: {
                        line: {
                            borderWidth: 3
                        }
                    },
                    scales: {
                        r: {
                            angleLines: {
                                display: true
                            },
                            suggestedMin: 0.5,
                            suggestedMax: 1.0
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Performance on Real-World Data'
                        }
                    }
                }
            });
            
            // Store reference globally
            window.realDataChart = this.charts.realDataChart;
            console.log('BenchmarkDashboard: Successfully created real data performance chart');
        } catch (error) {
            console.error('BenchmarkDashboard: Error creating real data performance chart:', error);
        }
    }
    
    /**
     * Initialize feature importance chart
     */
    initFeatureImportanceChart() {
        console.log('BenchmarkDashboard: Initializing feature importance chart');
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx || !this.data.featureImportance) {
            console.warn('BenchmarkDashboard: Cannot initialize feature importance chart - canvas or data missing');
            return;
        }
        
        // Ensure any existing chart is destroyed first
        try {
            if (window.featureImportanceChart && typeof window.featureImportanceChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing global feature importance chart');
                window.featureImportanceChart.destroy();
            } else if (window.featureImportanceChart) {
                console.warn('BenchmarkDashboard: Global feature importance chart exists but destroy method is not available');
                window.featureImportanceChart = null;
            }
            
            if (this.charts.featureImportanceChart && typeof this.charts.featureImportanceChart.destroy === 'function') {
                console.log('BenchmarkDashboard: Destroying existing instance feature importance chart');
                this.charts.featureImportanceChart.destroy();
            } else if (this.charts.featureImportanceChart) {
                console.warn('BenchmarkDashboard: Instance feature importance chart exists but destroy method is not available');
                this.charts.featureImportanceChart = null;
            }
        } catch (error) {
            console.error('BenchmarkDashboard: Error destroying existing feature importance chart:', error);
            // Reset chart references to avoid further issues
            window.featureImportanceChart = null;
            this.charts.featureImportanceChart = null;
        }
        
        try {
            // Get dataset selector
            const datasetSelector = document.getElementById('featureImportanceDataset');
            
            // Populate dataset selector
            if (datasetSelector) {
                datasetSelector.innerHTML = '';
                Object.keys(this.data.featureImportance).forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset;
                    option.textContent = dataset.replace('_', ' ').charAt(0).toUpperCase() + dataset.replace('_', ' ').slice(1);
                    datasetSelector.appendChild(option);
                });
                
                // Add event listener
                datasetSelector.addEventListener('change', () => this.updateFeatureImportanceChart());
            }
            
            // Get selected dataset or default to first one
            const selectedDataset = datasetSelector ? datasetSelector.value : Object.keys(this.data.featureImportance)[0];
            const datasetData = this.data.featureImportance[selectedDataset];
            
            // Get data from selected dataset
            const features = datasetData.features;
            const scores = datasetData.importance;
            
            // Create chart
            this.charts.featureImportanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: features,
                    datasets: [{
                        label: 'Importance',
                        data: scores,
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgb(75, 192, 192)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Importance'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `Feature Importance for ${selectedDataset.replace('_', ' ').charAt(0).toUpperCase() + selectedDataset.replace('_', ' ').slice(1)}`
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Importance: ${(context.raw * 100).toFixed(1)}%`;
                                }
                            }
                        }
                    }
                }
            });
            
            // Store reference globally
            window.featureImportanceChart = this.charts.featureImportanceChart;
        } catch (error) {
            console.error('BenchmarkDashboard: Error creating feature importance chart:', error);
        }
    }
    
    /**
     * Update feature importance chart when dataset changes
     */
    updateFeatureImportanceChart() {
        const datasetSelector = document.getElementById('featureImportanceDataset');
        if (!datasetSelector || !this.data.featureImportance) return;
        
        const selectedDataset = datasetSelector.value;
        const datasetData = this.data.featureImportance[selectedDataset];
        
        // Get data from selected dataset
        const features = datasetData.features;
        const scores = datasetData.importance;
        
        // Update chart
        if (this.charts.featureImportanceChart) {
            this.charts.featureImportanceChart.data.labels = features;
            this.charts.featureImportanceChart.data.datasets[0].data = scores;
            this.charts.featureImportanceChart.options.plugins.title.text = 
                `Feature Importance for ${selectedDataset.replace('_', ' ').charAt(0).toUpperCase() + selectedDataset.replace('_', ' ').slice(1)}`;
            this.charts.featureImportanceChart.update();
        } else {
            this.initFeatureImportanceChart();
        }
    }
    
    /**
     * Run benchmark comparison with synthetic data
     */
    async runBenchmarkComparison() {
        console.log('BenchmarkDashboard: Running benchmark comparison');
        
        // Show loading UI
        const button = document.getElementById('runBenchmarkComparison');
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Running...';
        button.disabled = true;
        
        try {
            // Call the API endpoint
            const response = await fetch('/api/dashboard/run-benchmark-comparison', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to run benchmark comparison');
            }
            
            const result = await response.json();
            console.log('BenchmarkDashboard: Benchmark comparison result:', result);
            
            // Show success message
            this.showSuccess('Successfully ran benchmark comparison with synthetic data');
            
            // Reload data to update charts
            await this.loadAllData();
            
        } catch (error) {
            console.error('BenchmarkDashboard: Error running benchmark comparison:', error);
            this.showError(`Failed to run benchmark comparison: ${error.message}`);
        } finally {
            // Restore button
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }
    
    /**
     * Show success message
     */
    showSuccess(message) {
        console.log(message);
        // Create and show alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
        alertDiv.style.zIndex = '9999';
        alertDiv.innerHTML = `
            <strong>Success!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        document.body.appendChild(alertDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
    
    /**
     * Show error message
     */
    showError(message) {
        console.error(message);
        // Create and show alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
        alertDiv.style.zIndex = '9999';
        alertDiv.innerHTML = `
            <strong>Error!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        document.body.appendChild(alertDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
    
    /**
     * Get color for an optimizer
     */
    getOptimizerColor(optimizer, alpha = 1) {
        const colors = {
            'Meta-Learner': `rgba(52, 152, 219, ${alpha})`,
            'ACO': `rgba(155, 89, 182, ${alpha})`,
            'Differential Evolution': `rgba(231, 76, 60, ${alpha})`,
            'Particle Swarm': `rgba(46, 204, 113, ${alpha})`,
            'Grey Wolf': `rgba(241, 196, 15, ${alpha})`,
            'CMA-ES': `rgba(52, 73, 94, ${alpha})`,
            'Bayesian': `rgba(230, 126, 34, ${alpha})`
        };
        
        return colors[optimizer] || `rgba(128, 128, 128, ${alpha})`;
    }
    
    /**
     * Get color for a metric
     */
    getMetricColor(metric, alpha = 1) {
        const colors = {
            'accuracy': `rgba(52, 152, 219, ${alpha})`,
            'precision': `rgba(46, 204, 113, ${alpha})`,
            'recall': `rgba(155, 89, 182, ${alpha})`,
            'f1': `rgba(241, 196, 15, ${alpha})`,
            'auc': `rgba(231, 76, 60, ${alpha})`,
            'specificity': `rgba(52, 73, 94, ${alpha})`
        };
        
        return colors[metric] || `rgba(128, 128, 128, ${alpha})`;
    }
    
    /**
     * Get fallback benchmark comparison data
     */
    getFallbackBenchmarkData() {
        const algorithms = ['meta-learner', 'aco', 'differential-evolution', 'particle-swarm', 'grey-wolf', 'bayesian'];
        const metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'training_time'];
        
        const result = {};
        metrics.forEach(metric => {
            result[metric] = {};
            algorithms.forEach(algo => {
                // Generate realistic fallback values
                let baseValue = 0;
                if (metric === 'training_time') {
                    baseValue = algo === 'meta-learner' ? 12.5 : Math.random() * 10 + 15;
                } else {
                    baseValue = algo === 'meta-learner' ? 0.93 : 0.85 + Math.random() * 0.08;
                }
                result[metric][algo] = baseValue;
            });
        });
        
        return result;
    }
    
    /**
     * Get fallback convergence data
     */
    getFallbackConvergenceData() {
        const algorithms = ['meta-learner', 'aco', 'differential-evolution', 'particle-swarm', 'grey-wolf', 'bayesian'];
        const iterations = Array.from({length: 50}, (_, i) => i + 1);
        
        const result = {};
        algorithms.forEach(algo => {
            result[algo] = {
                iterations: iterations,
                scores: iterations.map(i => {
                    // Generate realistic convergence curve
                    const maxScore = algo === 'meta-learner' ? 0.93 : 0.85 + Math.random() * 0.08;
                    return maxScore * (1 - Math.exp(-0.1 * i));
                })
            };
        });
        
        return result;
    }
    
    /**
     * Get fallback real data performance
     */
    getFallbackRealDataPerformance() {
        const algorithms = ['meta-learner', 'aco', 'differential-evolution', 'particle-swarm', 'grey-wolf', 'bayesian'];
        const metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc'];
        
        const result = {};
        metrics.forEach(metric => {
            result[metric] = {};
            algorithms.forEach(algo => {
                // Generate realistic fallback values
                const baseValue = algo === 'meta-learner' ? 0.92 : 0.84 + Math.random() * 0.08;
                result[metric][algo] = baseValue;
            });
        });
        
        return result;
    }
    
    /**
     * Get fallback feature importance data
     */
    getFallbackFeatureImportance() {
        return {
            "diabetes": {
                "features": ["glucose", "bmi", "age", "blood_pressure", "insulin"],
                "importance": [0.32, 0.24, 0.18, 0.15, 0.11]
            },
            "heart_disease": {
                "features": ["age", "cholesterol", "max_heart_rate", "st_depression", "chest_pain"],
                "importance": [0.29, 0.25, 0.21, 0.15, 0.10]
            }
        };
    }
}

// Will be instantiated by the unified dashboard
