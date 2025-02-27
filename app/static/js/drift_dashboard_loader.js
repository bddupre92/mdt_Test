/**
 * Drift Dashboard Loader
 * 
 * This script loads the drift visualization component and initializes it.
 */
document.addEventListener('DOMContentLoaded', function() {
    // Check if DriftVisualization class is already loaded
    if (typeof DriftVisualization === 'undefined') {
        // Load the drift visualization script
        const vizScript = document.createElement('script');
        vizScript.src = '/static/js/drift_visualization.js';
        
        // Initialize the dashboard after the script loads
        vizScript.onload = function() {
            initializeDashboard();
        };
        
        // Add the script to the document
        document.head.appendChild(vizScript);
    } else {
        // DriftVisualization is already loaded, initialize the dashboard
        initializeDashboard();
    }
    
    // Function to initialize the dashboard
    function initializeDashboard() {
        // Initialize the drift visualization dashboard
        const driftDashboard = new DriftVisualization('driftDashboardContainer');
        console.log('Drift detection dashboard initialized');
    }
});
