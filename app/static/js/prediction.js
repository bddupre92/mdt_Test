// Prediction UI functionality
class MigrainePredictionUI {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.loadHistory();
    }
    
    initializeElements() {
        // Form elements
        this.form = document.getElementById('prediction-form');
        this.sleepInput = document.getElementById('sleep-hours');
        this.stressInput = document.getElementById('stress-level');
        this.pressureInput = document.getElementById('weather-pressure');
        this.heartRateInput = document.getElementById('heart-rate');
        this.hormonalInput = document.getElementById('hormonal-level');
        
        // Results elements
        this.riskScore = document.getElementById('risk-score');
        this.riskLevel = document.getElementById('risk-level');
        this.triggersList = document.getElementById('triggers-list');
        
        // Chart elements
        this.historyChart = document.getElementById('history-chart');
        this.featureChart = document.getElementById('feature-importance-chart');
    }
    
    bindEvents() {
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.getPrediction();
        });
        
        // Real-time validation
        [this.sleepInput, this.stressInput, this.pressureInput,
         this.heartRateInput, this.hormonalInput].forEach(input => {
            input.addEventListener('input', () => this.validateInput(input));
        });
    }
    
    async getPrediction() {
        if (!this.validateForm()) return;
        
        const data = {
            sleep_hours: parseFloat(this.sleepInput.value),
            stress_level: parseInt(this.stressInput.value),
            weather_pressure: parseFloat(this.pressureInput.value),
            heart_rate: parseFloat(this.heartRateInput.value),
            hormonal_level: parseFloat(this.hormonalInput.value)
        };
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) throw new Error('Prediction failed');
            
            const result = await response.json();
            this.updateUI(result);
        } catch (error) {
            this.showError('Failed to get prediction. Please try again.');
        }
    }
    
    validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        if (isNaN(value) || value < min || value > max) {
            input.classList.add('invalid');
            return false;
        }
        
        input.classList.remove('invalid');
        return true;
    }
    
    validateForm() {
        let isValid = true;
        [this.sleepInput, this.stressInput, this.pressureInput,
         this.heartRateInput, this.hormonalInput].forEach(input => {
            if (!this.validateInput(input)) isValid = false;
        });
        return isValid;
    }
    
    updateUI(result) {
        // Update risk score and level
        this.riskScore.textContent = `${(result.risk_score * 100).toFixed(1)}%`;
        this.riskLevel.textContent = result.risk_level;
        this.riskLevel.className = `risk-level ${result.risk_level}`;
        
        // Update triggers list
        this.triggersList.innerHTML = result.top_triggers
            .map(trigger => `
                <li class="trigger-item ${trigger.status !== 'normal' ? 'warning' : ''}">
                    <span class="trigger-name">${trigger.name}</span>
                    <span class="trigger-value">${trigger.value}</span>
                    <span class="trigger-status">${trigger.status}</span>
                </li>
            `).join('');
            
        // Update feature importance chart
        this.updateFeatureChart(result.feature_importance);
    }
    
    async loadHistory() {
        try {
            const response = await fetch('/api/history');
            if (!response.ok) throw new Error('Failed to load history');
            
            const history = await response.json();
            this.updateHistoryChart(history);
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }
    
    updateHistoryChart(history) {
        const dates = history.map(h => h.date);
        const scores = history.map(h => h.risk_score);
        const occurrences = history.map(h => h.migraine_occurred ? 1 : 0);
        
        new Chart(this.historyChart, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Risk Score',
                    data: scores,
                    borderColor: '#4CAF50',
                    fill: false
                }, {
                    label: 'Migraine Occurred',
                    data: occurrences,
                    type: 'scatter',
                    backgroundColor: '#F44336'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    updateFeatureChart(importance) {
        const features = Object.keys(importance);
        const values = Object.values(importance);
        
        new Chart(this.featureChart, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Feature Importance',
                    data: values,
                    backgroundColor: '#2196F3'
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
    }
    
    showError(message) {
        // Implement error notification
        console.error(message);
        alert(message);
    }
}

// Initialize UI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.predictionUI = new MigrainePredictionUI();
});