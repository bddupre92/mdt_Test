const RISK_COLORS = {
    low: '#4CAF50',
    medium: '#FFC107',
    high: '#F44336'
};

class MigraineDashboard {
    constructor() {
        this.riskChart = null;
        this.triggerChart = null;
        this.driftChart = null;
        this.initializeCharts();
        this.setupEventListeners();
    }

    initializeCharts() {
        // Risk trend chart
        const riskCtx = document.getElementById('riskTrend').getContext('2d');
        this.riskChart = new Chart(riskCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Migraine Risk',
                    data: [],
                    borderColor: '#2196F3',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });

        // Top triggers chart
        const triggerCtx = document.getElementById('topTriggers').getContext('2d');
        this.triggerChart = new Chart(triggerCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Trigger Importance',
                    data: [],
                    backgroundColor: '#4CAF50'
                }]
            },
            options: {
                responsive: true,
                indexAxis: 'y'
            }
        });
    }

    setupEventListeners() {
        // Refresh data every 5 minutes
        setInterval(() => this.updateDashboard(), 300000);
        
        // Setup date range selector
        document.getElementById('dateRange').addEventListener('change', (e) => {
            this.updateDashboard(e.target.value);
        });

        // Setup Run Benchmarks button
        document.getElementById('runBenchmarks').addEventListener('click', () => {
            this.runBenchmarks();
        });
    }

    async updateDashboard(days = 30) {
        try {
            // Get prediction history
            const history = await this.fetchPredictionHistory(days);
            this.updateRiskChart(history);
            
            // Get feature importance
            const triggers = await this.fetchTriggerImportance();
            this.updateTriggerChart(triggers);
            
            // Update current risk display
            const currentRisk = history.predictions[history.predictions.length - 1];
            this.updateRiskDisplay(currentRisk);
            
            // Check for drift alerts
            const driftStatus = await this.fetchDriftStatus();
            this.updateDriftAlerts(driftStatus);
            
        } catch (error) {
            console.error('Error updating dashboard:', error);
            this.showError('Failed to update dashboard');
        }
    }

    async fetchPredictionHistory(days) {
        const response = await fetch(`/api/predictions/history?days=${days}`);
        if (!response.ok) throw new Error('Failed to fetch prediction history');
        return await response.json();
    }

    async fetchTriggerImportance() {
        const response = await fetch('/api/predictions/triggers');
        if (!response.ok) throw new Error('Failed to fetch trigger importance');
        return await response.json();
    }

    async fetchDriftStatus() {
        const response = await fetch('/api/predictions/drift-status');
        if (!response.ok) throw new Error('Failed to fetch drift status');
        return await response.json();
    }

    async runBenchmarks() {
        try {
            const response = await fetch('/api/dashboard/run-benchmark-comparison', {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to run benchmarks');
            const results = await response.json();
            this.displayBenchmarkResults(results);
        } catch (error) {
            console.error('Error running benchmarks:', error);
            this.showError('Failed to run benchmarks');
        }
    }

    displayBenchmarkResults(results) {
        const resultsSection = document.getElementById('benchmarkResults');
        resultsSection.innerHTML = '';

        results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.classList.add('result');
            resultDiv.innerHTML = `
                <h2>${result.optimizer_name}</h2>
                <p>Function: ${result.function_name}</p>
                <p>Dimension: ${result.dimension}</p>
                <p>Best Fitness: ${result.best_fitness}</p>
                <p>Mean Fitness: ${result.mean_fitness}</p>
                <p>Std Fitness: ${result.std_fitness}</p>
                <p>Iterations: ${result.iterations}</p>
                <p>Evaluations: ${result.evaluations}</p>
                <p>Time Taken: ${result.time_taken}</p>
                <p>Memory Peak: ${result.memory_peak}</p>
            `;
            resultsSection.appendChild(resultDiv);
        });
    }

    updateRiskChart(history) {
        this.riskChart.data.labels = history.dates;
        this.riskChart.data.datasets[0].data = history.predictions;
        this.riskChart.update();
    }

    updateTriggerChart(triggers) {
        this.triggerChart.data.labels = triggers.map(t => t.name);
        this.triggerChart.data.datasets[0].data = triggers.map(t => t.importance);
        this.triggerChart.update();
    }

    updateRiskDisplay(risk) {
        const display = document.getElementById('currentRisk');
        const riskLevel = risk < 0.3 ? 'low' : risk < 0.7 ? 'medium' : 'high';
        
        display.textContent = `${Math.round(risk * 100)}%`;
        display.style.color = RISK_COLORS[riskLevel];
        
        // Update risk factors
        const factorsDiv = document.getElementById('riskFactors');
        factorsDiv.innerHTML = this.formatRiskFactors(risk);
    }

    updateDriftAlerts(status) {
        const alertsDiv = document.getElementById('driftAlerts');
        if (status.drift_detected) {
            alertsDiv.innerHTML = `
                <div class="alert alert-warning">
                    Concept drift detected! Model is adapting to changes in your migraine patterns.
                    <ul>
                        ${status.changes.map(c => `<li>${c}</li>`).join('')}
                    </ul>
                </div>
            `;
        } else {
            alertsDiv.innerHTML = '';
        }
    }

    formatRiskFactors(risk) {
        return `
            <div class="risk-factor ${risk >= 0.7 ? 'high' : ''}">
                <i class="fas fa-exclamation-triangle"></i>
                High stress level detected
            </div>
            <div class="risk-factor ${risk >= 0.5 ? 'medium' : ''}">
                <i class="fas fa-cloud"></i>
                Weather pressure changes
            </div>
        `;
    }

    showError(message) {
        const errorDiv = document.getElementById('errorMessages');
        errorDiv.innerHTML = `
            <div class="alert alert-danger">
                ${message}
                <button type="button" class="close" data-dismiss="alert">&times;</button>
            </div>
        `;
    }
}

// Initialize dashboard when document is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new MigraineDashboard();
    window.dashboard.updateDashboard();
});
