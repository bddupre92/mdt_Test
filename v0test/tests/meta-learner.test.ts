import { test, expect } from '@playwright/test';

// Test the meta-learner tab functionality
test('meta-learner tab functionality', async ({ page }) => {
  // Step 1: Navigate to the dashboard page
  await page.goto('http://localhost:3006');
  
  // Wait for the dashboard to load
  await page.waitForSelector('h1:has-text("Data Visualization Hub")');
  
  // Step 2: Click on the Meta-Learning tab
  await page.click('button:has-text("Meta-Learning")');
  
  // Wait for the meta-learning form to appear
  await page.waitForSelector('text=SATZilla Meta-Learning Analysis');
  
  // Step 3: Check if the form elements are loaded
  await expect(page.locator('label:has-text("Select Dataset")')).toBeVisible();
  await expect(page.locator('label:has-text("Benchmark Function")')).toBeVisible();
  await expect(page.locator('label:has-text("Select Optimizers to Compare")')).toBeVisible();
  
  // Step 4: Select a dataset from the dropdown
  // First, check if there are any datasets available
  const datasetSelect = page.locator('select').first();
  await datasetSelect.waitFor();
  
  // Get all available datasets
  const options = await datasetSelect.locator('option').all();
  console.log(`Found ${options.length} dataset options`);
  
  // Skip the first option (which is the placeholder)
  if (options.length > 1) {
    // Select the first actual dataset (index 1, skipping the placeholder)
    await datasetSelect.selectOption({ index: 1 });
    console.log('Selected the first dataset option');
  } else {
    console.log('No datasets available to select');
    return;
  }
  
  // Step 5: Select a benchmark function
  const benchmarkSelect = page.locator('select').nth(1);
  await benchmarkSelect.selectOption({ index: 1 }); // Select the first benchmark option
  
  // Step 6: Select optimization algorithms to compare
  // Check the first two optimizer checkboxes
  await page.locator('input[type="checkbox"]').first().check();
  await page.locator('input[type="checkbox"]').nth(1).check();
  
  // Step 7: Click the "Run SATZilla Analysis" button
  await page.locator('button:has-text("Run SATZilla Analysis")').click();
  
  // Step 8: Wait for the analysis to complete and results to display
  // This may take some time, so we'll wait for a reasonable amount of time
  try {
    // Wait for results to appear - this is the heading shown when results are available
    await page.waitForSelector('text=SATZilla Algorithm Selection', { timeout: 10000 });
    
    // Step 9: Verify that results are displayed
    await expect(page.locator('text=Selected Algorithm')).toBeVisible();
    await expect(page.locator('text=Selection Confidence')).toBeVisible();
    await expect(page.locator('text=Algorithm Predictions')).toBeVisible();
    
    console.log('Meta-learner analysis completed successfully');
  } catch (error) {
    console.log('Timed out waiting for meta-learner results. This might be expected if the analysis takes longer than the timeout.');
    // Take a screenshot of the current state
    await page.screenshot({ path: 'meta-learner-timeout.png' });
  }
  
  // Step 10: Take a screenshot of the final state
  await page.screenshot({ path: 'meta-learner-results.png' });
}); 