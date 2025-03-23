import { defineConfig } from '@playwright/test';

/**
 * Playwright configuration for end-to-end testing
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  // Look for test files in the tests directory
  testDir: './tests',
  
  // Run tests in parallel
  fullyParallel: true,
  
  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,
  
  // Retry on CI only
  retries: process.env.CI ? 2 : 0,
  
  // Output test results
  reporter: [['html'], ['list']],
  
  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium',
      use: {
        // Browser viewport size
        viewport: { width: 1280, height: 720 },
        
        // Record screenshots and video
        screenshot: 'on',
        video: 'on-first-retry',
        
        // Record traces on retry
        trace: 'on-first-retry',
      },
    },
  ],
  
  // Server options
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3006',
    reuseExistingServer: !process.env.CI,
    timeout: 60 * 1000, // 60 seconds
  },
}); 