// Dashboard JavaScript for migraine prediction app
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
        const riskElement = document.getElementById('riskTrend');
        if (riskElement) {
            const riskCtx = riskElement.getContext('2d');
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
        } else {
            console.error('Risk trend chart element not found.');
        }

        // Top triggers chart
        const triggerElement = document.getElementById('topTriggers');
        if (triggerElement) {
            const triggerCtx = triggerElement.getContext('2d');
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
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        } else {
            console.error('Top triggers chart element not found.');
        }

        // Drift chart
        const driftElement = document.getElementById('driftChart');
        if (driftElement) {
            const driftCtx = driftElement.getContext('2d');
            this.driftChart = new Chart(driftCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Drift Data',
                        data: [],
                        borderColor: '#FFC107',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        } else {
            console.error('Drift chart element not found.');
        }
    }

    setupEventListeners() {
        // Refresh data every 5 minutes
        setInterval(() => this.updateDashboard(), 300000);
        
        // Setup date range selector
        document.getElementById('dateRange').addEventListener('change', (e) => {
            this.updateDashboard(e.target.value);
        });
    }

    updateDashboard() {
        this.fetchPatientData();
        this.fetchDriftResults();
        // Fetch drift detection data
        fetch('/api/dashboard/get_drift_data')
            .then(response => response.json())
            .then(data => {
                // Update risk chart
                this.riskChart.data.labels = data.labels;
                this.riskChart.data.datasets[0].data = data.risk_data;
                this.riskChart.update();

                // Update trigger chart
                this.triggerChart.data.labels = data.trigger_labels;
                this.triggerChart.data.datasets[0].data = data.trigger_data;
                this.triggerChart.update();

                // Update drift chart
                this.driftChart.data.labels = data.drift_labels;
                this.driftChart.data.datasets[0].data = data.drift_data;
                this.driftChart.update();
            })
            .catch(error => console.error('Error fetching data:', error));
    }

    async fetchPatientData() {
        try {
            const response = await fetch('/api/dashboard/get_patient_data');
            const data = await response.json();
            // Populate patient data section
            document.getElementById('patientData').innerHTML = `<p>${data.patient_info}</p>`;
        } catch (error) {
            console.error('Error fetching patient data:', error);
        }
    }

    async fetchDriftResults() {
        try {
            const response = await fetch('/api/dashboard/get_drift_results');
            const data = await response.json();
            // Populate drift results section
            document.getElementById('driftResults').innerHTML = `<p>${data.drift_info}</p>`;
        } catch (error) {
            console.error('Error fetching drift results:', error);
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
