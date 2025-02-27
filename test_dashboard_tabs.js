// Test script to verify dashboard tab loading
(function() {
    // Wait for DOM to be fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        console.log('===== DASHBOARD TESTING SCRIPT =====');
        
        // Get all tab buttons
        const tabButtons = document.querySelectorAll('#dashboardTabs button');
        console.log(`Found ${tabButtons.length} dashboard tabs`);
        
        // Function to click each tab with a delay
        function clickTabWithDelay(index) {
            if (index >= tabButtons.length) {
                console.log('===== ALL TABS TESTED =====');
                return;
            }
            
            const tab = tabButtons[index];
            console.log(`Testing tab: ${tab.textContent.trim()}`);
            
            // Click the tab
            tab.click();
            
            // Wait 3 seconds before clicking the next tab
            setTimeout(() => {
                clickTabWithDelay(index + 1);
            }, 3000);
        }
        
        // Start testing tabs after 2 seconds
        setTimeout(() => {
            clickTabWithDelay(0);
        }, 2000);
    });
})();
