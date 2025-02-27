document.addEventListener('DOMContentLoaded', function() {
    const runBenchmarksButton = document.getElementById('runBenchmarksButton');
    const resultsSection = document.getElementById('resultsSection');

    runBenchmarksButton.addEventListener('click', function() {
        fetch('/api/dashboard/run-benchmark-comparison', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Benchmark comparison triggered:', data);
            fetchBenchmarkResults();
        })
        .catch(error => {
            console.error('Error triggering benchmark comparison:', error);
        });
    });

    function fetchBenchmarkResults() {
        fetch('/api/benchmarks')
        .then(response => response.json())
        .then(data => {
            displayBenchmarkResults(data);
        })
        .catch(error => {
            console.error('Error fetching benchmark results:', error);
        });
    }

    function displayBenchmarkResults(results) {
        resultsSection.innerHTML = '';

        if (results.length === 0) {
            resultsSection.innerHTML = '<p>No benchmark results available.</p>';
            return;
        }

        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');

        const headers = ['Optimizer', 'Function', 'Dimension', 'Best Fitness', 'Mean Fitness', 'Std Fitness', 'Iterations', 'Evaluations', 'Time Taken', 'Memory Peak'];
        const headerRow = document.createElement('tr');
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        results.forEach(result => {
            const row = document.createElement('tr');
            Object.values(result).forEach(value => {
                const td = document.createElement('td');
                td.textContent = value;
                row.appendChild(td);
            });
            tbody.appendChild(row);
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        resultsSection.appendChild(table);

        // Create a chart to visualize benchmark results
        const chartContainer = document.createElement('div');
        chartContainer.classList.add('chart-container');
        const chartCanvas = document.createElement('canvas');
        chartContainer.appendChild(chartCanvas);
        resultsSection.appendChild(chartContainer);

        const ctx = chartCanvas.getContext('2d');
        const chartData = {
            labels: results.map(result => result.Function),
            datasets: [{
                label: 'Best Fitness',
                data: results.map(result => result['Best Fitness']),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        };

        new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Fetch initial benchmark results on page load
    fetchBenchmarkResults();
});
