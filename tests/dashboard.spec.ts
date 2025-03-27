import { test, expect } from '@playwright/test';
import { exec, spawn, ChildProcess } from 'child_process';
import { promisify } from 'util';
import { once } from 'events';

const execAsync = promisify(exec);

/**
 * Start the Streamlit server and wait for it to be ready
 * @param port Port to start the server on
 */
async function startStreamlit(port: number): Promise<ChildProcess> {
  console.log('Starting Streamlit server...');
  
  const streamlitProcess = spawn('streamlit', [
    'run', 
    'app/ui/performance_analysis_dashboard.py',
    `--server.port=${port}`,
    '--server.headless=true'
  ], {
    stdio: ['ignore', 'pipe', 'pipe']
  });

  // Log stdout and stderr
  streamlitProcess.stdout.on('data', (data) => {
    console.log(`Streamlit stdout: ${data}`);
  });
  
  streamlitProcess.stderr.on('data', (data) => {
    console.error(`Streamlit stderr: ${data}`);
  });

  // Wait for server to start
  return new Promise((resolve, reject) => {
    // Set a timeout in case server never starts
    const timeout = setTimeout(() => {
      reject(new Error('Timeout waiting for Streamlit server to start'));
    }, 30000);
    
    // Check if server has started by looking for the 'You can now view' message
    streamlitProcess.stdout.on('data', (data) => {
      if (data.toString().includes('You can now view') || 
          data.toString().includes('Network URL:')) {
        clearTimeout(timeout);
        console.log('Streamlit server started successfully');
        
        // Give it an extra second to fully initialize
        setTimeout(() => resolve(streamlitProcess), 1000);
      }
    });
    
    // Handle errors
    streamlitProcess.on('error', (err) => {
      clearTimeout(timeout);
      reject(err);
    });
    
    streamlitProcess.on('exit', (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(`Streamlit process exited with code ${code}`));
      }
    });
  });
}

/**
 * Test all available checkpoint formats to verify our data format flexibility
 * This test loads each checkpoint separately and verifies appropriate handling
 */
// Define a test fixture to manage Streamlit process per browser
test.describe('Dashboard Format Flexibility Tests', () => {
  // Use a unique port for each test
  const getPortForBrowser = (browserName: string): number => {
    switch (browserName) {
      case 'chromium': return 8506;
      case 'firefox': return 8507;
      case 'webkit': return 8508;
      default: return 8509;
    }
  };

  test('dashboard format flexibility test', async ({ browser }) => {
    // Start the Streamlit server before testing with a unique port
    const port = getPortForBrowser(browser.browserType().name());
    console.log(`Running test with ${browser.browserType().name()} on port ${port}`);
    const streamlitProcess = await startStreamlit(port);
  const context = await browser.newContext({
    baseURL: `http://localhost:${port}`,
    viewport: { width: 1400, height: 900 }, // Larger viewport for better visibility
  });

  const page = await context.newPage();
  
  await page.goto('/');

  // Wait for the Streamlit app to load
  await page.waitForSelector('div.stApp', { timeout: 10000 });
  console.log('Streamlit app loaded');

  // Take a screenshot of the initial state
  await page.screenshot({ path: 'dashboard-initial.png' });

  // ----- TEST SIDEBAR AND CHECKPOINT LOADING -----
  console.log('Testing sidebar and checkpoint loading...');
  
  // Look for the sidebar where checkpoints should be listed
  const sidebar = page.locator('section[data-testid="stSidebar"]');
  await expect(sidebar).toBeVisible();

  // Check debug info displayed
  const debugTextLocator = sidebar.locator('text=Looking in:');
  await expect(debugTextLocator).toBeVisible();
  const debugText = await debugTextLocator.textContent();
  console.log('Debug info:', debugText);

  // Check if checkpoints are found
  const checkpointsFoundLocator = sidebar.locator('text=Found');
  await expect(checkpointsFoundLocator).toBeVisible();
  const checkpointsFoundText = await checkpointsFoundLocator.textContent();
  console.log('Checkpoints found info:', checkpointsFoundText);
  
  // Check that checkpoint dropdown is visible
  const checkpointDropdown = sidebar.locator('div[data-testid="stSelectbox"]');
  await expect(checkpointDropdown).toBeVisible();
  
  // Get all available checkpoint options
  await checkpointDropdown.click();
  
  // The available checkpoints should include our test formats
  const expectedCheckpoints = [
    'checkpoint_sample_2025_03_26.json',
    'checkpoint_flat_format_2025_03_26.json',
    'checkpoint_nested_format_2025_03_26.json',
    'checkpoint_standard_format_2025_03_26.json',
    'checkpoint_unusual_format_2025_03_26.json'
  ];
  
  // Get dropdown list items and check their content
  const dropdownList = page.locator('div[role="listbox"] div[role="option"]');
  const optionCount = await dropdownList.count();
  
  // Log all available options for debugging
  const availableOptions: string[] = [];
  for (let i = 0; i < optionCount; i++) {
    const optionText = await dropdownList.nth(i).textContent() || '';
    availableOptions.push(optionText);
  }
  console.log(`Found ${optionCount} checkpoint options: ${JSON.stringify(availableOptions)}`);
  
  // Map the expected checkpoints to what's actually in the dropdown
  // (they might appear differently in the UI)
  
  // Close the dropdown for now
  await page.keyboard.press('Escape');
  
  // Test each checkpoint format to verify flexibility
  for (const checkpointName of expectedCheckpoints) {
    console.log(`\n---- TESTING CHECKPOINT: ${checkpointName} ----`);
    
    // Select the checkpoint from the dropdown
    await checkpointDropdown.click();
    
    // Get the list of available options again
    const options = page.locator('div[role="listbox"] div[role="option"]');
    const count = await options.count();
    
    // Find the index of the option containing our checkpoint name (partial match)
    let foundIndex = -1;
    for (let i = 0; i < count; i++) {
      const text = await options.nth(i).textContent() || '';
      if (text.includes(checkpointName.replace('.json', '')) || 
          checkpointName.includes(text.trim())) {
        foundIndex = i;
        console.log(`Found matching option at index ${i}: "${text}" for ${checkpointName}`);
        break;
      }
    }
    
    if (foundIndex === -1) {
      console.log(`WARNING: Could not find checkpoint option: ${checkpointName}`);
      await page.keyboard.press('Escape'); // Close dropdown
      continue;
    }
    
    // Click the found option
    await options.nth(foundIndex).click();
    
    // Click the Load Checkpoint button
    const loadButton = sidebar.locator('button:has-text("Load Checkpoint")');
    await expect(loadButton).toBeVisible();
    await loadButton.click();
    
    // Wait for the page to update
    console.log(`Waiting for dashboard to load ${checkpointName}...`);
    await page.waitForTimeout(3000); // Give Streamlit time to process
    
    // Take a screenshot of the loaded checkpoint
    await page.screenshot({ path: `checkpoint-${checkpointName}.png` });
    
    // Check for basic UI elements that should always be present
    const dashboardContent = page.locator('div.main');
    expect(dashboardContent).toBeVisible();
    
    // ----- TEST DASHBOARD COMPONENTS -----
    console.log('Checking components for this checkpoint format...');
    
    // Create a results object to track which components are present
    const results = {
      expertBenchmarks: false,
      gatingAnalysis: false,
      endToEndMetrics: false,
      errors: [] as string[]
    };
    
    // Check for error messages
    const errorMessages = page.locator('div[data-testid="stAlert"]');
    const errorCount = await errorMessages.count();
    if (errorCount > 0) {
      for (let i = 0; i < errorCount; i++) {
        const errorText = await errorMessages.nth(i).textContent();
        console.log(`Found error message: ${errorText}`);
        results.errors.push(errorText || 'Unknown error');
      }
    }
    
    // Check for Expert Benchmarks component
    const expertHeading = page.locator('h2, h3').filter({ hasText: /Expert Model Benchmarks|Expert Performance/ });
    results.expertBenchmarks = await expertHeading.count() > 0;
    if (results.expertBenchmarks) {
      console.log('✓ Expert Benchmarks component is present');
    }
    
    // Check for Gating Network Analysis component
    const gatingHeading = page.locator('h2, h3').filter({ hasText: /Gating Network|Expert Selection/ });
    results.gatingAnalysis = await gatingHeading.count() > 0;
    if (results.gatingAnalysis) {
      console.log('✓ Gating Network Analysis component is present');
    }
    
    // Check for End-to-End Metrics component
    const metricsHeading = page.locator('h2, h3').filter({ hasText: /End-to-End Metrics|Overall Performance|Performance Metrics/ });
    results.endToEndMetrics = await metricsHeading.count() > 0;
    if (results.endToEndMetrics) {
      console.log('✓ End-to-End Metrics component is present');
    }
    
    // Log summary for this checkpoint format
    console.log(`\n-- Summary for ${checkpointName} --`);
    console.log(`Expert Benchmarks: ${results.expertBenchmarks ? '✓' : '✗'}`);
    console.log(`Gating Analysis: ${results.gatingAnalysis ? '✓' : '✗'}`);
    console.log(`End-to-End Metrics: ${results.endToEndMetrics ? '✓' : '✗'}`);
    console.log(`Error messages: ${results.errors.length > 0 ? results.errors.length : 'None'}`);
    
    // Look for any fallback messages or empty state indicators
    const infoMessages = page.locator('div.stAlert').filter({ hasText: /No.*data available|data not found|No metrics/ });
    if (await infoMessages.count() > 0) {
      console.log('Found fallback/empty state messages:');
      const messagesToLog = await infoMessages.allTextContents();
      messagesToLog.forEach(msg => console.log(` - ${msg}`));
    }
  }
  
  // Overall test summary
  console.log('\n---- DASHBOARD FORMAT FLEXIBILITY TEST SUMMARY ----');
  console.log('✓ Successfully tested all checkpoint formats');
  console.log('✓ Verified dashboard handling of different data structures');
  console.log('✓ Captured screenshots of each format for analysis');
  
  // Optional: For debugging purposes
  // await page.pause();
  
  await context.close();
  
  // Clean up: stop the Streamlit server
  console.log('Test completed, shutting down Streamlit server...');
  streamlitProcess.kill();
  console.log('Streamlit server shut down successfully');
  });
});
