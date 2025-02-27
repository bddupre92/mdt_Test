// Simple logging script to monitor dashboard initialization
(function() {
    console.log('==== Dashboard Test Logger Initialized ====');
    
    // Capture and enhance console.error
    const originalConsoleError = console.error;
    console.error = function() {
        // Add a prefix to make errors more visible
        const args = Array.from(arguments);
        args.unshift('⚠️ ERROR:');
        originalConsoleError.apply(console, args);
    };
    
    // Watch for tab changes
    document.addEventListener('DOMContentLoaded', function() {
        const tabButtons = document.querySelectorAll('#dashboardTabs button');
        
        tabButtons.forEach(tab => {
            tab.addEventListener('click', function() {
                console.log(`📋 Tab clicked: ${this.textContent.trim()}`);
            });
        });
        
        // Check if charts are initialized
        setInterval(function() {
            console.log('📊 Chart Status:');
            console.log('  - Drift Chart:', window.driftSeverityChart ? 'Initialized' : 'Not initialized');
            console.log('  - Performance Chart:', window.performanceChart ? 'Initialized' : 'Not initialized');
            console.log('  - Benchmark Comparison Chart:', window.benchmarkComparisonChart ? 'Initialized' : 'Not initialized');
            console.log('  - Convergence Chart:', window.convergenceChart ? 'Initialized' : 'Not initialized');
            console.log('  - Real Data Chart:', window.realDataChart ? 'Initialized' : 'Not initialized');
            console.log('  - Feature Importance Chart:', window.featureImportanceChart ? 'Initialized' : 'Not initialized');
        }, 5000);
    });
})();
