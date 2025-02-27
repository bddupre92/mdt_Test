/**
 * Unified Dashboard JavaScript
 * 
 * This file contains all the JavaScript code for the unified dashboard,
 * including drift detection, performance metrics, optimization, and meta-analysis.
 */

class UnifiedDashboard {
    constructor() {
        // Store data
        this.data = {
            drift: null,
            performance: null,
            optimization: null,
            benchmarks: null,
            metaAnalysis: null
        };
        
        // Set initial active tab
        const activeTabButton = document.querySelector('#dashboardTabs button.active');
        if (activeTabButton) {
            const tabId = activeTabButton.id;
            this.activeTab = '#' + tabId.replace('-tab', '') + '-tab-content';
        } else {
            // Default to drift tab if no active tab
            this.activeTab = '#drift-tab-content';
        }
        console.log('UnifiedDashboard: Initial active tab set to:', this.activeTab);
        
        // Initialize the dashboard
        this.initialize();
    }
    
    /**
     * Initialize the dashboard
     */
    async initialize() {
        console.log('UnifiedDashboard: Initializing...');
        
        try {
            // Register event handlers
            this.registerEventHandlers();
            console.log('UnifiedDashboard: Event handlers registered');
            
            // Initialize BenchmarkDashboard instance
            this.benchmarkDashboard = new BenchmarkDashboard(this);
            console.log('UnifiedDashboard: BenchmarkDashboard created');
            
            // Load data for the active tab
            console.log('UnifiedDashboard: Active tab is:', this.activeTab);
            
            // Load data for all tabs at startup for a better experience
            console.log('UnifiedDashboard: Loading all data for better user experience');
            
            // Load drift data
            console.log('UnifiedDashboard: Loading drift data');
            await this.loadDriftData();
            
            // Load performance metrics
            console.log('UnifiedDashboard: Loading performance metrics');
            await this.loadPerformanceMetrics();
            
            // Load optimization data
            console.log('UnifiedDashboard: Loading optimization data');
            await this.loadOptimizationData();
            
            // Load benchmark data
            console.log('UnifiedDashboard: Loading benchmark data through BenchmarkDashboard');
            await this.loadBenchmarkData();
            
            // Load meta-analysis data
            console.log('UnifiedDashboard: Loading meta-analysis data');
            await this.loadMetaAnalysisData();
            
            console.log('UnifiedDashboard: Initialization successful');
        } catch (error) {
            console.error('UnifiedDashboard: Error during initialization:', error);
            this.showError('Failed to initialize dashboard: ' + error.message);
        } finally {
            // Hide loading overlay
            this.showLoading(false);
        }
    }
    
    /**
     * Register event handlers for all interactive elements
     */
    registerEventHandlers() {
        console.log('UnifiedDashboard: Registering event handlers');
        
        // Tab change event
        const tabs = document.querySelectorAll('#dashboardTabs button[data-bs-toggle="tab"]');
        console.log('UnifiedDashboard: Found ' + tabs.length + ' tabs');
        
        tabs.forEach(tab => {
            tab.addEventListener('shown.bs.tab', (event) => {
                console.log('UnifiedDashboard: Tab changed to:', event.target.id);
                this.activeTab = '#' + event.target.id.replace('-tab', '') + '-content';
                
                // Lazy-load data for the tab if not already loaded
                if (event.target.id === 'drift-tab' && !this.data.drift) {
                    console.log('UnifiedDashboard: Lazy-loading drift data');
                    this.loadDriftData();
                } else if (event.target.id === 'performance-tab' && !this.data.performance) {
                    console.log('UnifiedDashboard: Lazy-loading performance data');
                    this.loadPerformanceMetrics();
                } else if (event.target.id === 'optimization-tab' && !this.data.optimization) {
                    console.log('UnifiedDashboard: Lazy-loading optimization data');
                    this.loadOptimizationData();
                } else if (event.target.id === 'benchmarks-tab' && !this.data.benchmarks) {
                    console.log('UnifiedDashboard: Lazy-loading benchmark data');
                    this.loadBenchmarkData();
                } else if (event.target.id === 'meta-tab' && !this.data.metaAnalysis) {
                    console.log('UnifiedDashboard: Lazy-loading meta-analysis data');
                    this.loadMetaAnalysisData();
                }
                
                // Resize charts if needed
                window.dispatchEvent(new Event('resize'));
            });
        });
        
        // Set up optimization button handlers using event delegation
        document.addEventListener('click', (e) => {
            const button = e.target.closest('button');
            if (!button) return;
            
            if (button.classList.contains('optimizer-btn')) {
                const optimizerType = button.dataset.optimizer;
                console.log('UnifiedDashboard: Optimizer button clicked:', optimizerType);
                this.runOptimization(optimizerType);
            } else if (button.id === 'runMetaOptimization') {
                console.log('UnifiedDashboard: Meta optimization button clicked');
                this.runMetaOptimization();
            } else if (button.id === 'runBenchmarkComparison') {
                console.log('UnifiedDashboard: Benchmark comparison button clicked');
                this.runBenchmarkComparison();
            }
        });
        
        // Set up refresh interval for dynamic data
        setInterval(() => {
            const activeTab = document.querySelector('#dashboardTabs button.active');
            if (activeTab) {
                const tabId = activeTab.id;
                console.log('UnifiedDashboard: Periodic refresh check for tab:', tabId);
                if (tabId === 'performance-tab') {
                    this.loadPerformanceMetrics();
                } else if (tabId === 'drift-tab') {
                    this.loadDriftData();
                }
            }
        }, 60000); // Refresh every minute
        
        console.log('UnifiedDashboard: Event handlers registered successfully');
    }
    
    /**
     * Load drift detection data
     */
    async loadDriftData() {
        try {
            this.showLoading(true);
            
            // Use the sample data endpoint for testing
            const response = await fetch('/api/dashboard/sample-drift-data');
            if (!response.ok) {
                throw new Error('Failed to load drift data');
            }
            
            this.data.drift = await response.json();
            console.log('Loaded drift data:', this.data.drift);
            
            // Initialize drift charts
            this.initializeDriftCharts();
        } catch (error) {
            console.error('Error loading drift data:', error);
            this.showError('Failed to load drift data: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Load and display performance metrics
     */
    async loadPerformanceMetrics() {
        try {
            this.showLoading(true);
            
            // Fetch performance metrics data
            const response = await fetch('/api/dashboard/performance');
            if (!response.ok) {
                throw new Error('Failed to fetch performance metrics');
            }
            const data = await response.json();
            
            // Update metrics cards
            const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc'];
            metrics.forEach(metric => {
                const element = document.getElementById(`${metric}-value`);
                if (element && data.current_metrics[metric] !== undefined) {
                    element.textContent = data.current_metrics[metric].toFixed(3);
                }
            });
            
            // Create performance over time chart
            const ctx = document.getElementById('performanceChart');
            if (ctx) {
                if (this.performanceChart) {
                    this.performanceChart.destroy();
                }
                this.performanceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.history.timestamps,
                        datasets: metrics.map(metric => ({
                            label: metric.replace('_', ' ').toUpperCase(),
                            data: data.history[metric],
                            borderColor: this.getMetricColor(metric),
                            fill: false
                        }))
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Performance Metrics Over Time'
                            }
                        }
                    }
                });
            }
            
            // Create radar chart for current metrics
            const radarCtx = document.getElementById('metricsRadarChart');
            if (radarCtx) {
                if (this.radarChart) {
                    this.radarChart.destroy();
                }
                this.radarChart = new Chart(radarCtx, {
                    type: 'radar',
                    data: {
                        labels: metrics.map(m => m.replace('_', ' ').toUpperCase()),
                        datasets: [{
                            label: 'Current Metrics',
                            data: metrics.map(m => data.current_metrics[m]),
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            borderColor: 'rgb(54, 162, 235)',
                            pointBackgroundColor: 'rgb(54, 162, 235)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgb(54, 162, 235)'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        elements: {
                            line: {
                                borderWidth: 3
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Current Performance Metrics'
                            }
                        }
                    }
                });
            }
        } catch (error) {
            console.error('Error loading performance metrics:', error);
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Load and display optimization data
     */
    async loadOptimizationData() {
        console.log('UnifiedDashboard: loadOptimizationData called');
        this.showLoading(true);
        
        try {
            // Fetch optimization data from API
            const [optimizationResults, optimizationHistory] = await Promise.all([
                this.fetchData('/api/dashboard/optimization-results'),
                this.fetchData('/api/dashboard/optimization-history')
            ]);
            
            // Store data
            this.data.optimization = {
                results: optimizationResults,
                history: optimizationHistory
            };
            
            console.log('UnifiedDashboard: Optimization data loaded:', this.data.optimization);
            
            // Destroy existing charts
            this.destroyCharts('optimization');
            
            // Initialize charts
            this.initOptimizationProgressChart();
            this.initHyperparameterImportanceChart();
            
            this.showSuccess('Optimization data loaded successfully');
        } catch (error) {
            console.error('Error loading optimization data:', error);
            this.showError(`Failed to load optimization data: ${error.message}`);
            
            // Generate fallback data if needed
            if (!this.data.optimization) {
                this.data.optimization = this.getFallbackOptimizationData();
                
                // Try to initialize charts with fallback data
                this.destroyCharts('optimization');
                this.initOptimizationProgressChart();
                this.initHyperparameterImportanceChart();
            }
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Load meta-analysis data
     */
    async loadMetaAnalysisData() {
        console.log('UnifiedDashboard: Loading meta-analysis data...');
        this.showLoading(true);
        
        try {
            const data = await this.fetchData('/api/dashboard/meta-analysis');
            if (!data) {
                throw new Error('Failed to load meta-analysis data');
            }
            
            // Store data
            this.data.metaAnalysis = data;
            
            console.log('UnifiedDashboard: Meta-analysis data loaded:', this.data.metaAnalysis);
            
            // Destroy existing charts
            this.destroyCharts('metaAnalysis');
            
            // Initialize charts
            this.initMetaAnalysisCharts();
            
            this.showSuccess('Meta-analysis data loaded successfully');
        } catch (error) {
            console.error('Error loading meta-analysis data:', error);
            this.showError('Failed to load meta-analysis data: ' + error.message);
            
            // Generate fallback data
            this.data.metaAnalysis = this.getFallbackMetaAnalysisData();
            
            // Try to initialize charts with fallback data
            this.destroyCharts('metaAnalysis');
            this.initMetaAnalysisCharts();
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Destroy charts for a specific module to prevent Chart.js errors
     */
    destroyCharts(module) {
        console.log(`UnifiedDashboard: Destroying ${module} charts`);
        
        switch(module) {
            case 'optimization':
                try {
                    if (window.optimizationProgressChart && typeof window.optimizationProgressChart.destroy === 'function') {
                        console.log('UnifiedDashboard: Destroying existing optimization progress chart');
                        window.optimizationProgressChart.destroy();
                    } else if (window.optimizationProgressChart) {
                        console.warn('UnifiedDashboard: Optimization progress chart exists but destroy method is not available');
                    }
                    window.optimizationProgressChart = null;
                    
                    if (window.hyperparameterImportanceChart && typeof window.hyperparameterImportanceChart.destroy === 'function') {
                        console.log('UnifiedDashboard: Destroying existing hyperparameter importance chart');
                        window.hyperparameterImportanceChart.destroy();
                    } else if (window.hyperparameterImportanceChart) {
                        console.warn('UnifiedDashboard: Hyperparameter importance chart exists but destroy method is not available');
                    }
                    window.hyperparameterImportanceChart = null;
                } catch (error) {
                    console.error('UnifiedDashboard: Error destroying optimization charts:', error);
                    window.optimizationProgressChart = null;
                    window.hyperparameterImportanceChart = null;
                }
                break;
                
            case 'metaAnalysis':
                try {
                    if (window.metaAnalysisChart && typeof window.metaAnalysisChart.destroy === 'function') {
                        window.metaAnalysisChart.destroy();
                    } else if (window.metaAnalysisChart) {
                        console.warn('UnifiedDashboard: Meta analysis chart exists but destroy method is not available');
                    }
                    window.metaAnalysisChart = null;
                    
                    if (window.metaRecommendationChart && typeof window.metaRecommendationChart.destroy === 'function') {
                        window.metaRecommendationChart.destroy();
                    } else if (window.metaRecommendationChart) {
                        console.warn('UnifiedDashboard: Meta recommendation chart exists but destroy method is not available');
                    }
                    window.metaRecommendationChart = null;
                } catch (error) {
                    console.error('UnifiedDashboard: Error destroying meta analysis charts:', error);
                    window.metaAnalysisChart = null;
                    window.metaRecommendationChart = null;
                }
                break;
                
            case 'drift':
                try {
                    if (window.driftSeverityChart && typeof window.driftSeverityChart.destroy === 'function') {
                        window.driftSeverityChart.destroy();
                    } else if (window.driftSeverityChart) {
                        console.warn('UnifiedDashboard: Drift severity chart exists but destroy method is not available');
                    }
                    window.driftSeverityChart = null;
                    
                    if (window.featureDistributionChart && typeof window.featureDistributionChart.destroy === 'function') {
                        window.featureDistributionChart.destroy();
                    } else if (window.featureDistributionChart) {
                        console.warn('UnifiedDashboard: Feature distribution chart exists but destroy method is not available');
                    }
                    window.featureDistributionChart = null;
                } catch (error) {
                    console.error('UnifiedDashboard: Error destroying drift charts:', error);
                    window.driftSeverityChart = null;
                    window.featureDistributionChart = null;
                }
                break;
                
            default:
                console.warn(`UnifiedDashboard: Unknown module '${module}' for chart destruction`);
        }
    }
    
    /**
     * Initialize optimization progress chart
     */
    initOptimizationProgressChart() {
        console.log('UnifiedDashboard: Initializing optimization progress chart');
        const ctx = document.getElementById('optimizationProgressChart');
        if (!ctx || !this.data.optimization?.history) {
            console.warn('UnifiedDashboard: Cannot initialize optimization progress chart. Canvas or data missing.');
            return;
        }
        
        // Ensure chart is destroyed first - this should be handled by destroyCharts, but adding an extra check
        try {
            if (window.optimizationProgressChart && typeof window.optimizationProgressChart.destroy === 'function') {
                console.log('UnifiedDashboard: Destroying existing optimization progress chart');
                window.optimizationProgressChart.destroy();
            } else if (window.optimizationProgressChart) {
                console.warn('UnifiedDashboard: Optimization progress chart exists but destroy method is not available');
            }
            window.optimizationProgressChart = null;
        } catch (error) {
            console.error('UnifiedDashboard: Error destroying optimization progress chart:', error);
            window.optimizationProgressChart = null;
        }
        
        // Extract data from optimization history
        const history = this.data.optimization.history;
        
        try {
            // Create chart
            console.log('UnifiedDashboard: Creating new optimization progress chart');
            window.optimizationProgressChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: history.iterations,
                    datasets: [{
                        label: 'Optimization Score (AUC)',
                        data: history.scores,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: Math.max(0, Math.min(...history.scores) - 0.05),
                            max: Math.min(1, Math.max(...history.scores) + 0.05),
                            title: {
                                display: true,
                                text: 'AUC Score'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Optimization Progress'
                        }
                    }
                }
            });
            console.log('UnifiedDashboard: Successfully created optimization progress chart');
        } catch (error) {
            console.error('UnifiedDashboard: Error creating optimization progress chart:', error);
        }
    }
    
    /**
     * Initialize hyperparameter importance chart
     */
    initHyperparameterImportanceChart() {
        console.log('UnifiedDashboard: Initializing hyperparameter importance chart');
        const ctx = document.getElementById('hyperparameterImportanceChart');
        if (!ctx || !this.data.optimization?.results?.parameter_importance) {
            console.warn('UnifiedDashboard: Cannot initialize hyperparameter importance chart. Canvas or data missing.');
            return;
        }
        
        // Ensure chart is destroyed first - this should be handled by destroyCharts, but adding an extra check
        try {
            if (window.hyperparameterImportanceChart && typeof window.hyperparameterImportanceChart.destroy === 'function') {
                console.log('UnifiedDashboard: Destroying existing hyperparameter importance chart');
                window.hyperparameterImportanceChart.destroy();
            } else if (window.hyperparameterImportanceChart) {
                console.warn('UnifiedDashboard: Hyperparameter importance chart exists but destroy method is not available');
            }
            window.hyperparameterImportanceChart = null;
        } catch (error) {
            console.error('UnifiedDashboard: Error destroying hyperparameter importance chart:', error);
            window.hyperparameterImportanceChart = null;
        }
        
        // Extract data
        const paramImportance = this.data.optimization.results.parameter_importance;
        const params = Object.keys(paramImportance);
        const values = Object.values(paramImportance);
        
        try {
            // Create chart
            console.log('UnifiedDashboard: Creating new hyperparameter importance chart');
            window.hyperparameterImportanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: params,
                    datasets: [{
                        label: 'Importance',
                        data: values,
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgb(54, 162, 235)',
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
                            text: 'Hyperparameter Importance'
                        }
                    }
                }
            });
            console.log('UnifiedDashboard: Successfully created hyperparameter importance chart');
        } catch (error) {
            console.error('UnifiedDashboard: Error creating hyperparameter importance chart:', error);
        }
    }
    
    /**
     * Initialize meta-analysis charts
     */
    initMetaAnalysisCharts() {
        console.log('UnifiedDashboard: Initializing meta-analysis charts');
        this.initModelComparisonChart();
        this.initMetaRecommendationChart();
    }
    
    /**
     * Initialize model comparison chart for meta-analysis
     */
    initModelComparisonChart() {
        const ctx = document.getElementById('modelComparisonChart');
        if (!ctx || !this.data.metaAnalysis?.model_comparison) return;
        
        // Ensure chart is destroyed first
        if (window.metaAnalysisChart) {
            window.metaAnalysisChart.destroy();
        }
        
        // Extract data
        const comparison = this.data.metaAnalysis.model_comparison;
        const models = Object.keys(comparison);
        const metrics = ['accuracy', 'f1_score', 'precision', 'recall'];
        
        // Create datasets for each metric
        const datasets = metrics.map(metric => {
            return {
                label: metric.replace('_', ' ').toUpperCase(),
                data: models.map(model => comparison[model][metric]),
                backgroundColor: this.getColorForMetric(metric, 0.7)
            };
        });
        
        // Create chart
        window.metaAnalysisChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: models.map(m => m.toUpperCase()),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        min: 0.5,
                        max: 1,
                        ticks: {
                            stepSize: 0.1
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Comparison'
                    }
                }
            }
        });
    }
    
    /**
     * Initialize meta recommendation chart
     */
    initMetaRecommendationChart() {
        const ctx = document.getElementById('metaRecommendationChart');
        if (!ctx || !this.data.metaAnalysis?.optimizer_recommendations) return;
        
        // Ensure chart is destroyed first
        if (window.metaRecommendationChart) {
            window.metaRecommendationChart.destroy();
        }
        
        // Extract data
        const recommendations = this.data.metaAnalysis.optimizer_recommendations;
        const optimizers = Object.keys(recommendations);
        const counts = Object.values(recommendations);
        
        // Create chart
        window.metaRecommendationChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: optimizers.map(o => o.toUpperCase()),
                datasets: [{
                    label: 'Recommendation Count',
                    data: counts,
                    backgroundColor: 'rgba(123, 104, 238, 0.7)',
                    borderColor: 'rgb(123, 104, 238)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Optimizer Recommendation Frequency'
                    }
                }
            }
        });
    }
    
    /**
     * Get color for specific metric
     */
    getColorForMetric(metric, alpha = 1) {
        const colors = {
            accuracy: `rgba(75, 192, 192, ${alpha})`,
            f1_score: `rgba(153, 102, 255, ${alpha})`,
            precision: `rgba(255, 159, 64, ${alpha})`,
            recall: `rgba(54, 162, 235, ${alpha})`,
            auc: `rgba(255, 99, 132, ${alpha})`
        };
        
        return colors[metric] || `rgba(201, 203, 207, ${alpha})`;
    }
    
    /**
     * Generate fallback optimization data
     */
    getFallbackOptimizationData() {
        console.log('UnifiedDashboard: Generating fallback optimization data');
        
        // Generate some realistic looking optimization history
        const iterations = Array.from({length: 20}, (_, i) => i + 1);
        const baseScore = 0.75;
        const scores = iterations.map(i => {
            // Simulate gradual improvement with diminishing returns
            return baseScore + 0.2 * (1 - Math.exp(-i / 10));
        });
        
        // Generate timestamps
        const now = new Date();
        const timestamps = iterations.map(i => {
            const date = new Date(now);
            date.setHours(date.getHours() - (20 - i));
            return date.toLocaleTimeString();
        });
        
        // Generate parameter importance
        const paramImportance = {
            'learning_rate': 0.32,
            'max_depth': 0.27,
            'min_samples_split': 0.18,
            'min_samples_leaf': 0.13,
            'n_estimators': 0.10
        };
        
        // Generate optimization results
        const results = {
            'best_score': scores[scores.length - 1],
            'best_params': {
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'n_estimators': 200
            },
            'parameter_importance': paramImportance,
            'training_time': 345.6,
            'algorithm': 'Meta-Learner'
        };
        
        return {
            results: results,
            history: {
                iterations: timestamps,
                scores: scores
            }
        };
    }
    
    /**
     * Generate fallback meta-analysis data
     */
    getFallbackMetaAnalysisData() {
        console.log('UnifiedDashboard: Generating fallback meta-analysis data');
        
        // Models comparison data
        const modelComparison = {
            'meta_learner': {
                'accuracy': 0.88,
                'f1_score': 0.87,
                'precision': 0.86,
                'recall': 0.89,
                'auc': 0.92
            },
            'ensemble': {
                'accuracy': 0.85,
                'f1_score': 0.83,
                'precision': 0.84,
                'recall': 0.82,
                'auc': 0.88
            },
            'gradient_boosting': {
                'accuracy': 0.82,
                'f1_score': 0.81,
                'precision': 0.83,
                'recall': 0.80,
                'auc': 0.86
            },
            'random_forest': {
                'accuracy': 0.80,
                'f1_score': 0.79,
                'precision': 0.81,
                'recall': 0.78,
                'auc': 0.84
            }
        };
        
        // Optimizer recommendations
        const optimizerRecommendations = {
            'de_adaptive': 4,
            'cma-es': 3,
            'aco': 3,
            'gwo': 1, 
            'de_standard': 1,
            'pso': 1,
            'bayesian': 1
        };
        
        // Feature interactions
        const featureInteractions = [
            {'feature1': 'stress_level', 'feature2': 'sleep_quality', 'interaction_strength': 0.72},
            {'feature1': 'weather_changes', 'feature2': 'barometric_pressure', 'interaction_strength': 0.68},
            {'feature1': 'caffeine_intake', 'feature2': 'hydration', 'interaction_strength': 0.57},
            {'feature1': 'screen_time', 'feature2': 'stress_level', 'interaction_strength': 0.51},
            {'feature1': 'physical_activity', 'feature2': 'sleep_quality', 'interaction_strength': 0.48}
        ];
        
        return {
            model_comparison: modelComparison,
            optimizer_recommendations: optimizerRecommendations,
            feature_interactions: featureInteractions
        };
    }
    
    /**
     * Load benchmark data
     */
    async loadBenchmarkData() {
        console.log('UnifiedDashboard: loadBenchmarkData called');
        
        try {
            this.showLoading(true);
            
            // Initialize BenchmarkDashboard if not already done
            if (!this.benchmarkDashboard) {
                console.log('UnifiedDashboard: Initializing BenchmarkDashboard');
                this.benchmarkDashboard = new BenchmarkDashboard(this);
            }
            
            // Use the BenchmarkDashboard instance to load data
            console.log('UnifiedDashboard: Calling BenchmarkDashboard.loadData()');
            
            // Check if the benchmarkDashboard is properly initialized
            if (this.benchmarkDashboard && typeof this.benchmarkDashboard.loadData === 'function') {
                await this.benchmarkDashboard.loadData();
                this.data.benchmarks = true; // Mark as loaded
                console.log('UnifiedDashboard: Benchmark data loaded successfully');
            } else {
                console.error('UnifiedDashboard: BenchmarkDashboard not properly initialized');
                throw new Error('BenchmarkDashboard not properly initialized');
            }
        } catch (error) {
            console.error('UnifiedDashboard: Error loading benchmark data:', error);
            this.showError('Failed to load benchmark data: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Render benchmark charts
     */
    renderBenchmarkCharts() {
        if (!this.data.benchmarks) return;
        
        // Benchmark comparison chart
        this.renderBenchmarkComparisonChart();
        
        // Convergence chart
        this.renderConvergenceChart();
        
        // Real data performance chart
        this.renderRealDataPerformanceChart();
        
        // Feature importance chart
        this.renderFeatureImportanceChart();
        
        // Set up event listeners for chart updates
        const convergenceSelect = document.getElementById('convergenceFunction');
        if (convergenceSelect) {
            convergenceSelect.addEventListener('change', () => this.renderConvergenceChart());
        }
        
        const datasetSelect = document.getElementById('datasetSelector');
        if (datasetSelect) {
            datasetSelect.addEventListener('change', () => this.renderRealDataPerformanceChart());
        }
    }
    
    /**
     * Render benchmark comparison chart
     */
    renderBenchmarkComparisonChart() {
        const ctx = document.getElementById('benchmarkComparisonChart');
        if (!ctx || !this.data.benchmarks.comparison) return;
        
        // Destroy existing chart if it exists
        if (this.charts.benchmarkComparison) {
            this.charts.benchmarkComparison.destroy();
        }
        
        // Prepare data
        const functions = Object.keys(this.data.benchmarks.comparison);
        const optimizers = Object.keys(this.data.benchmarks.comparison[functions[0]] || {});
        
        // Generate datasets
        const datasets = optimizers.map(optimizer => {
            const color = this.getOptimizerColor(optimizer);
            return {
                label: optimizer,
                data: functions.map(func => this.data.benchmarks.comparison[func][optimizer].mean),
                backgroundColor: color.replace(')', ', 0.7)').replace('rgb', 'rgba'),
                borderColor: color,
                borderWidth: 1
            };
        });
        
        // Create chart
        this.charts.benchmarkComparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: functions,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Performance'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Benchmark Function'
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
    }
    
    /**
     * Render convergence chart
     */
    renderConvergenceChart() {
        const ctx = document.getElementById('convergenceChart');
        if (!ctx || !this.data.benchmarks.convergence) return;
        
        // Get selected function
        const select = document.getElementById('convergenceFunction');
        const selectedFunction = select ? select.value : 'rosenbrock';
        
        // Destroy existing chart if it exists
        if (this.charts.convergence) {
            this.charts.convergence.destroy();
        }
        
        // Get data for optimizers
        const optimizers = Object.keys(this.data.benchmarks.convergence);
        
        // Create datasets
        const datasets = optimizers.map(optimizer => {
            const color = this.getOptimizerColor(optimizer);
            return {
                label: optimizer,
                data: this.data.benchmarks.convergence[optimizer].performance,
                borderColor: color,
                backgroundColor: color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
                fill: false,
                tension: 0.3
            };
        });
        
        // Create chart
        this.charts.convergence = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.data.benchmarks.convergence[optimizers[0]].iterations,
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
    }
    
    /**
     * Render real data performance chart
     */
    renderRealDataPerformanceChart() {
        const ctx = document.getElementById('realDataPerformanceChart');
        if (!ctx || !this.data.benchmarks.realData) return;
        
        // Get selected dataset
        const select = document.getElementById('datasetSelector');
        const selectedDataset = select ? select.value : 'Migraine_Combined';
        
        // Destroy existing chart if it exists
        if (this.charts.realDataPerformance) {
            this.charts.realDataPerformance.destroy();
        }
        
        // Get data for the selected dataset
        const datasetData = this.data.benchmarks.realData[selectedDataset];
        if (!datasetData) {
            console.error('No data for selected dataset:', selectedDataset);
            return;
        }
        
        // Get models and metrics
        const models = Object.keys(datasetData);
        const metrics = Object.keys(datasetData[models[0]]);
        
        // Create datasets (one dataset per metric)
        const datasets = metrics.map((metric, index) => {
            const color = this.getMetricColor(metric);
            return {
                label: metric.toUpperCase(),
                data: models.map(model => datasetData[model][metric]),
                backgroundColor: color.replace(')', ', 0.7)').replace('rgb', 'rgba'),
                borderColor: color,
                borderWidth: 1,
            };
        });
        
        // Create chart
        this.charts.realDataPerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Model'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Performance on ${selectedDataset.replace('_', ' ')} Dataset`
                    }
                }
            }
        });
    }
    
    /**
     * Render feature importance chart
     */
    renderFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx || !this.data.benchmarks.features) return;
        
        // Destroy existing chart if it exists
        if (this.charts.featureImportance) {
            this.charts.featureImportance.destroy();
        }
        
        // Get data
        const features = this.data.benchmarks.features.features;
        const scores = this.data.benchmarks.features.importance_scores;
        
        // Create chart
        this.charts.featureImportance = new Chart(ctx, {
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
                            text: 'Importance Score'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Importance'
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    /**
     * Get color for an optimizer
     */
    getOptimizerColor(optimizer) {
        const colors = {
            'Meta-Learner': 'rgb(52, 152, 219)',
            'ACO': 'rgb(155, 89, 182)',
            'Differential Evolution': 'rgb(231, 76, 60)',
            'Particle Swarm': 'rgb(46, 204, 113)',
            'Grey Wolf': 'rgb(241, 196, 15)',
            'CMA-ES': 'rgb(52, 73, 94)',
            'Bayesian': 'rgb(230, 126, 34)'
        };
        
        return colors[optimizer] || 'rgb(128, 128, 128)';
    }
    
    /**
     * Get color for a metric
     */
    getMetricColor(metric) {
        const colors = {
            'accuracy': 'rgb(52, 152, 219)',
            'precision': 'rgb(46, 204, 113)',
            'recall': 'rgb(155, 89, 182)',
            'f1': 'rgb(241, 196, 15)',
            'auc': 'rgb(231, 76, 60)',
            'specificity': 'rgb(52, 73, 94)'
        };
        
        return colors[metric] || 'rgb(128, 128, 128)';
    }
    
    /**
     * Run an optimization algorithm
     * @param {string} optimizerType - The type of optimizer to run
     */
    async runOptimization(optimizerType) {
        try {
            if (!optimizerType) {
                throw new Error('No optimizer type specified');
            }
            
            this.showLoading(true);
            console.log(`Starting ${optimizerType} optimization...`);
            
            const response = await fetch('/api/dashboard/run-optimization', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ optimizer_type: optimizerType })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Failed to run ${optimizerType} optimization`);
            }
            
            const result = await response.json();
            console.log(`${optimizerType} optimization result:`, result);
            
            // Show success message
            this.showSuccess(`Successfully started ${optimizerType} optimization`);
            
            // Refresh the optimization data
            await this.loadOptimizationData();
            
            // Make the optimization tab active if it's not already
            const optimizationTab = document.getElementById('optimization-tab');
            if (optimizationTab && !optimizationTab.classList.contains('active')) {
                optimizationTab.click();
            }
            
        } catch (error) {
            console.error(`Error running ${optimizerType} optimization:`, error);
            this.showError(`Failed to run ${optimizerType} optimization: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Run meta-optimization
     */
    async runMetaOptimization() {
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/dashboard/run-meta-optimization', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to run meta-optimization');
            }
            
            const result = await response.json();
            console.log('Meta-optimization started:', result);
            
            // Refresh the optimization data
            await this.loadOptimizationData();
            
        } catch (error) {
            console.error('Error running meta-optimization:', error);
            this.showError(`Failed to run meta-optimization: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Run benchmark comparison with synthetic data
     */
    async runBenchmarkComparison() {
        console.log('UnifiedDashboard: Running benchmark comparison');
        this.showLoading(true);
        
        try {
            // Set a timeout to ensure loading doesn't get stuck
            const loadingTimeout = setTimeout(() => {
                console.warn('UnifiedDashboard: Benchmark comparison timed out');
                this.showLoading(false);
                this.showError('Benchmark comparison timed out. The server may be unavailable.');
            }, 20000); // 20 second timeout
            
            // Call the API
            const response = await fetch('/api/dashboard/run-benchmark-comparison', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            // Clear timeout
            clearTimeout(loadingTimeout);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            console.log('UnifiedDashboard: Benchmark comparison result:', result);
            
            if (result.status === 'success') {
                // Reload benchmark data to update the charts
                await this.loadBenchmarkData();
                this.showSuccess('Benchmark comparison completed successfully');
            } else {
                throw new Error(result.message || 'Failed to run benchmark comparison');
            }
        } catch (error) {
            console.error('UnifiedDashboard: Error running benchmark comparison:', error);
            this.showError(`Failed to run benchmark comparison: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Show success message
     */
    showSuccess(message) {
        console.log('UnifiedDashboard: Success:', message);
        
        // Create and show alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
        alertDiv.setAttribute('role', 'alert');
        alertDiv.style.zIndex = '9999';
        alertDiv.style.maxWidth = '80%';
        alertDiv.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        
        alertDiv.innerHTML = `
            <strong>Success!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Automatically remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.classList.remove('show');
                setTimeout(() => alertDiv.remove(), 500);
            }
        }, 5000);
    }
    
    /**
     * Show error message
     */
    showError(message) {
        console.error('UnifiedDashboard: Error:', message);
        
        // Create and show alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
        alertDiv.setAttribute('role', 'alert');
        alertDiv.style.zIndex = '9999';
        alertDiv.style.maxWidth = '80%';
        alertDiv.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        
        alertDiv.innerHTML = `
            <strong>Error!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Automatically remove after 10 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.classList.remove('show');
                setTimeout(() => alertDiv.remove(), 500);
            }
        }, 10000);
    }
    
    /**
     * Format date for display
     */
    formatDate(timestamp) {
        const date = new Date(timestamp * 1000);
        return `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear()} ${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
    }
    
    /**
     * Initialize drift detection charts
     */
    initializeDriftCharts() {
        if (!this.data.drift) return;
        
        // Drift Severity Chart
        this.createDriftSeverityChart();
        
        // Feature Drift Chart
        this.createFeatureDriftChart();
        
        // Drift Trend Chart
        this.createDriftTrendChart();
    }
    
    /**
     * Create drift severity chart
     */
    createDriftSeverityChart() {
        const ctx = document.getElementById('driftSeverityChart')?.getContext('2d');
        if (!ctx) return;
        
        // Format dates for x-axis
        const labels = this.data.drift.timestamps.map(ts => this.formatDate(ts));
        
        // Create annotations for drift points
        const annotations = {};
        this.data.drift.drift_points.forEach((point, index) => {
            annotations[`line${index}`] = {
                type: 'line',
                mode: 'vertical',
                scaleID: 'x',
                value: point,
                borderColor: 'rgba(255, 0, 0, 0.7)',
                borderWidth: 2,
                label: {
                    content: 'Drift Detected',
                    enabled: true,
                    position: 'top'
                }
            };
        });
        
        // Destroy existing chart if it exists
        if (this.charts.driftSeverity) {
            this.charts.driftSeverity.destroy();
        }
        
        // Create new chart
        this.charts.driftSeverity = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Drift Severity',
                    data: this.data.drift.severities,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    annotation: {
                        annotations: annotations
                    },
                    title: {
                        display: true,
                        text: 'Drift Severity Over Time'
                    },
                    tooltip: {
                        callbacks: {
                            title: (tooltipItems) => {
                                return this.formatDate(this.data.drift.timestamps[tooltipItems[0].dataIndex]);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Severity'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Create feature drift chart
     */
    createFeatureDriftChart() {
        const ctx = document.getElementById('featureDriftChart')?.getContext('2d');
        if (!ctx) return;
        
        const featureNames = Object.keys(this.data.drift.feature_drifts);
        const featureCounts = Object.values(this.data.drift.feature_drifts);
        
        // Destroy existing chart if it exists
        if (this.charts.featureDrift) {
            this.charts.featureDrift.destroy();
        }
        
        // Create new chart
        this.charts.featureDrift = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: featureNames,
                datasets: [{
                    label: 'Drift Count',
                    data: featureCounts,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Drift Distribution'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Create drift trend chart
     */
    createDriftTrendChart() {
        const ctx = document.getElementById('driftTrendChart')?.getContext('2d');
        if (!ctx) return;
        
        // Format dates for x-axis
        const labels = this.data.drift.timestamps.map(ts => this.formatDate(ts));
        
        // Calculate trend (simple moving average)
        const windowSize = 5;
        const trends = [];
        
        for (let i = 0; i < this.data.drift.severities.length; i++) {
            if (i < windowSize - 1) {
                trends.push(null);
            } else {
                let sum = 0;
                for (let j = 0; j < windowSize; j++) {
                    sum += this.data.drift.severities[i - j];
                }
                trends.push(sum / windowSize);
            }
        }
        
        // Destroy existing chart if it exists
        if (this.charts.driftTrend) {
            this.charts.driftTrend.destroy();
        }
        
        // Create new chart
        this.charts.driftTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Severity',
                    data: this.data.drift.severities,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    tension: 0.1
                }, {
                    label: 'Trend (MA)',
                    data: trends,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Drift Severity Trend Analysis'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Severity'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Initialize performance charts
     */
    initializePerformanceCharts() {
        // Placeholder for performance charts
        // Will be implemented when the API is available
        console.log('Performance charts not yet implemented');
    }
    
    /**
     * Initialize optimization charts
     */
    initializeOptimizationCharts() {
        if (!this.data.optimization) {
            console.error('No optimization data available');
            return;
        }
        
        this.createOptimizerPerformanceChart();
        this.createHyperparameterImportanceChart();
        this.createOptimizationProgressChart();
        
        // Set up event listeners for optimization buttons
        document.getElementById('runMetaLearner')?.addEventListener('click', () => {
            this.runOptimization('meta-learner');
        });
        
        document.getElementById('runACO')?.addEventListener('click', () => {
            this.runOptimization('aco');
        });
    }
    
    /**
     * Create optimizer performance chart
     */
    createOptimizerPerformanceChart() {
        const ctx = document.getElementById('optimizationChart')?.getContext('2d');
        if (!ctx) return;
        
        // Extract data from history
        const history = this.data.optimization.history;
        const optimizers = history.optimizers || [];
        const performances = history.avg_performances || [];
        
        // Create the chart
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: optimizers,
                datasets: [{
                    label: 'Average Performance',
                    data: performances,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Optimizer Performance Comparison'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Performance: ${context.raw.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Performance Score'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Create hyperparameter importance chart
     */
    createHyperparameterImportanceChart() {
        const ctx = document.getElementById('hyperparameterChart')?.getContext('2d');
        if (!ctx) return;
        
        // Use the first optimizer's features as hyperparameters
        const history = this.data.optimization.history;
        const firstOptimizer = history.optimizers?.[0] || 'Unknown';
        const features = history.features?.[firstOptimizer] || {};
        
        // Extract feature names and values
        const featureNames = Object.keys(features);
        const featureValues = Object.values(features);
        
        // Create the chart
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: featureNames,
                datasets: [{
                    label: 'Feature Values',
                    data: featureValues,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Problem Feature Analysis'
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0
                    }
                }
            }
        });
    }
    
    /**
     * Create optimization progress chart
     */
    createOptimizationProgressChart() {
        const ctx = document.getElementById('optimizationProgressChart')?.getContext('2d');
        if (!ctx) return;
        
        // Extract success rates from history
        const history = this.data.optimization.history;
        const optimizers = history.optimizers || [];
        const successRates = optimizers.map(opt => history.success_rates?.[opt] || 0);
        
        // Create the chart
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: optimizers,
                datasets: [{
                    label: 'Success Rate (%)',
                    data: successRates,
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(153, 102, 255, 0.5)',
                        'rgba(255, 159, 64, 0.5)',
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(255, 205, 86, 0.5)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(255, 99, 132, 1)',
                        'rgba(255, 205, 86, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Optimizer Success Rates'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Success Rate: ${context.raw.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Initialize meta-analysis charts
     */
    initializeMetaAnalysisCharts() {
        if (!this.data.metaAnalysis) {
            console.error('No meta-analysis data available');
            return;
        }
        
        this.createMetaAnalysisOverviewChart();
        this.createModelComparisonChart();
        this.createFeatureImportanceChart();
    }
    
    /**
     * Create meta-analysis overview chart
     */
    createMetaAnalysisOverviewChart() {
        const ctx = document.getElementById('metaAnalysisChart')?.getContext('2d');
        if (!ctx) return;
        
        // Extract problem characteristics
        const characteristics = this.data.metaAnalysis.problem_characteristics || {};
        const labels = Object.keys(characteristics);
        
        // Map text values to numeric values for visualization
        const valueMap = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'present': 3,
            'absent': 1
        };
        
        const values = Object.values(characteristics).map(val => valueMap[val] || 2);
        
        // Create the chart
        new Chart(ctx, {
            type: 'polarArea',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Problem Characteristics',
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 205, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)',
                        'rgba(255, 159, 64, 0.5)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Problem Characteristics Analysis'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = characteristics[label] || '';
                                return `${label}: ${value}`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Create model comparison chart
     */
    createModelComparisonChart() {
        const ctx = document.getElementById('modelComparisonChart')?.getContext('2d');
        if (!ctx) return;
        
        // Extract optimizer recommendations
        const recommendations = this.data.metaAnalysis.optimizer_recommendations || {};
        const categories = Object.keys(recommendations);
        
        // Count the number of recommendations for each optimizer
        const optimizerCounts = {};
        
        for (const category of categories) {
            const optimizers = recommendations[category] || [];
            for (const optimizer of optimizers) {
                optimizerCounts[optimizer] = (optimizerCounts[optimizer] || 0) + 1;
            }
        }
        
        const optimizers = Object.keys(optimizerCounts);
        const counts = Object.values(optimizerCounts);
        
        // Create the chart
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: optimizers,
                datasets: [{
                    label: 'Recommendation Count',
                    data: counts,
                    backgroundColor: 'rgba(153, 102, 255, 0.5)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Optimizer Recommendation Frequency'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Create feature importance chart
     */
    createFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart')?.getContext('2d');
        if (!ctx) return;
        
        // Extract feature importance
        const featureImportance = this.data.metaAnalysis.feature_importance || {};
        const features = Object.keys(featureImportance);
        const importanceValues = Object.values(featureImportance);
        
        // Create the chart
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: features,
                datasets: [{
                    label: 'Importance',
                    data: importanceValues,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 205, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)',
                        'rgba(255, 159, 64, 0.5)',
                        'rgba(201, 203, 207, 0.5)',
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 205, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)',
                        'rgba(201, 203, 207, 0.7)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Importance'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const percentage = (value * 100).toFixed(1);
                                return `${context.label}: ${percentage}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update optimization results table
     */
    updateOptimizationResults(result) {
        const tbody = document.getElementById('optimizationResultsBody');
        if (!tbody) return;
        
        // Create a new row
        const row = document.createElement('tr');
        
        // Format best parameters as a string
        const bestParams = result.best_params ? 
            Object.entries(result.best_params)
                .map(([key, value]) => `${key}: ${value}`)
                .join('<br>') :
            'N/A';
        
        row.innerHTML = `
            <td>${result.algorithm}</td>
            <td>${result.performance ? result.performance.toFixed(3) : 'N/A'}</td>
            <td>${result.iterations || 'N/A'}</td>
            <td>${result.time_taken ? result.time_taken.toFixed(1) : 'N/A'}</td>
            <td>${bestParams}</td>
            <td>
                <span class="badge ${result.success ? 'bg-success' : 'bg-danger'}">
                    ${result.success ? 'Success' : 'Failed'}
                </span>
            </td>
        `;
        
        // Add the new row at the top
        if (tbody.firstChild) {
            tbody.insertBefore(row, tbody.firstChild);
        } else {
            tbody.appendChild(row);
        }
        
        // Limit to last 5 results
        while (tbody.children.length > 5) {
            tbody.removeChild(tbody.lastChild);
        }
    }
    
    /**
     * Update meta-learner results table
     */
    updateMetaLearnerResults(result) {
        const tbody = document.getElementById('metaLearnerResultsBody');
        if (!tbody) return;
        
        // Clear existing results
        tbody.innerHTML = '';
        
        // Create row for meta-learner results
        const row = document.createElement('tr');
        
        // Format best parameters as a string
        const bestParams = result.best_params ? 
            Object.entries(result.best_params)
                .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
                .join('<br>') :
            'N/A';
        
        row.innerHTML = `
            <td>${result.algorithm}</td>
            <td>${result.auc.toFixed(3)}</td>
            <td>${result.iterations}</td>
            <td>${result.time_taken.toFixed(1)}</td>
            <td>${bestParams}</td>
            <td>
                <span class="badge ${result.success ? 'bg-success' : 'bg-danger'}">
                    ${result.success ? 'Success' : 'Failed'}
                </span>
            </td>
        `;
        
        tbody.appendChild(row);
    }
    
    /**
     * Update optimizer recommendations
     */
    updateOptimizerRecommendations(recommendations) {
        const container = document.getElementById('recommendationsContainer');
        if (!container) return;
        
        container.innerHTML = recommendations.map(rec => `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h6>${rec.name}</h6>
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" 
                                style="width: ${rec.confidence * 100}%" 
                                aria-valuenow="${rec.confidence * 100}" 
                                aria-valuemin="0" 
                                aria-valuemax="100">
                                ${(rec.confidence * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    /**
     * Update AUC comparison chart
     */
    updateAucComparisonChart(results) {
        const ctx = document.getElementById('aucComparisonChart');
        if (!ctx) return;
        
        if (this.aucChart) {
            this.aucChart.destroy();
        }
        
        this.aucChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Meta-Learner', 'ACO'],
                datasets: [{
                    label: 'AUC Score',
                    data: [results.meta_learner.auc, results.aco.auc],
                    backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(255, 99, 132, 0.5)'],
                    borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    /**
     * Update time comparison chart
     */
    updateTimeComparisonChart(results) {
        const ctx = document.getElementById('timeComparisonChart');
        if (!ctx) return;
        
        if (this.timeChart) {
            this.timeChart.destroy();
        }
        
        this.timeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Meta-Learner', 'ACO'],
                datasets: [
                    {
                        label: 'Execution Time (s)',
                        data: [results.meta_learner.time_taken, results.aco.time_taken],
                        backgroundColor: ['rgba(75, 192, 192, 0.5)', 'rgba(153, 102, 255, 0.5)'],
                        borderColor: ['rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)'],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    /**
     * Show or hide loading overlay
     */
    showLoading(show) {
        console.log(`UnifiedDashboard: ${show ? 'Showing' : 'Hiding'} loading overlay`);
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            // Handle both the display style and the Bootstrap d-none class
            if (show) {
                loadingOverlay.style.display = 'flex';
                loadingOverlay.classList.remove('d-none');
            } else {
                loadingOverlay.style.display = 'none';
                loadingOverlay.classList.add('d-none');
            }
            console.log(`UnifiedDashboard: Loading overlay ${show ? 'shown' : 'hidden'}`);
        } else {
            console.warn('UnifiedDashboard: Loading overlay element not found');
        }
    }
    
    /**
     * Fetch data from an API endpoint with error handling
     * @param {string} url - The API endpoint URL
     * @returns {Promise<object>} - The parsed JSON response
     */
    async fetchData(url) {
        console.log(`UnifiedDashboard: Fetching data from ${url}`);
        try {
            const response = await fetch(url);
            if (!response.ok) {
                const errorText = await response.text();
                console.error(`UnifiedDashboard: HTTP error ${response.status} from ${url}:`, errorText);
                throw new Error(`HTTP error ${response.status}: ${errorText.substring(0, 100)}`);
            }
            
            const data = await response.json();
            console.log(`UnifiedDashboard: Successfully fetched data from ${url}`);
            return data;
        } catch (error) {
            console.error(`UnifiedDashboard: Error fetching from ${url}:`, error);
            
            // Enhanced error handling
            if (error.name === 'AbortError') {
                console.warn(`UnifiedDashboard: Request to ${url} timed out`);
                throw new Error(`Request to ${url.split('/').pop()} timed out. The server may be unavailable.`);
            }
            if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
                console.warn(`UnifiedDashboard: Network error when fetching from ${url}`);
                throw new Error(`Network error when fetching ${url.split('/').pop()}. Check if the server is running.`);
            }
            if (error.message.includes('SyntaxError')) {
                console.warn(`UnifiedDashboard: Invalid JSON response from ${url}`);
                throw new Error(`Invalid response format from ${url.split('/').pop()}. The server may be returning invalid data.`);
            }
            
            // For any other errors
            throw new Error(`Error fetching ${url.split('/').pop()}: ${error.message}`);
        }
    }
}

// Initialize the dashboard when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new UnifiedDashboard();
});
