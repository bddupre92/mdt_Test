# Comprehensive Testing Guide for Migraine DT

This guide explains how to run comprehensive tests for all functionalities in the Migraine DT application.

## What's Included

Our test suite provides comprehensive coverage for:

1. **Basic Page Navigation** - Verifies the datasets page loads correctly
2. **Tab Navigation** - Tests navigation between Explore, Upload, and Generate tabs
3. **API Endpoints** - Validates all API endpoints work correctly:
   - `/api/datasets` - Dataset listing endpoint
   - `/api/datasets/[id]/statistics` - Dataset statistics endpoint
   - `/api/framework/*` - Framework-related endpoints

## Setup

1. Make sure you have Node.js and npm installed
2. Install project dependencies:
   ```bash
   npm install
   ```
3. Install Playwright and browser dependencies:
   ```bash
   npx playwright install --with-deps
   ```

## Running Tests

### Option 1: Quick Run (Recommended)

We've created a test script that automatically checks if your server is running, runs all tests, and provides a summary:

```bash
# Make sure you're in the project root
cd /path/to/migraine-dt/v0test

# Run the comprehensive test script
./scripts/test-all-features.sh
```

### Option 2: Manual Testing

If you prefer to run tests manually:

1. Start the development server (if not already running):
   ```bash
   npm run dev
   ```

2. Run the tests:
   ```bash
   # Run all tests on all browsers
   npm test
   
   # Run tests on a specific browser
   npx playwright test --project=chromium
   
   # Run tests with UI mode for debugging
   npm run test:ui
   ```

3. View test results:
   ```bash
   # Open test report
   npx playwright show-report
   ```

## Test Organization

The test suite is organized to be resilient to UI changes:

1. **Basic Navigation**: Simple tests that verify the page loads correctly
2. **Tab Navigation**: Tests that the tab structure works properly
3. **API Tests**: Verify all backend endpoints are functioning correctly

## Troubleshooting

If tests fail:

1. **API Endpoint Errors**: Make sure the development server is running
2. **Browser Issues**: Try running with just one browser using `--project=chromium`
3. **Trace Files**: Examine trace files for failures using `npx playwright show-trace <path-to-trace>`
4. **HTML Report**: View the HTML report using `npx playwright show-report`

## Extending the Tests

To add new tests:

1. Add new test cases to `v0test/tests/datasets-page.spec.ts`
2. Follow the pattern of using resilient selectors
3. Include appropriate error handling for UI components that might change
4. Consider adding API tests for any new endpoints

## Continuous Integration

These tests are designed to work well in CI environments. You can run them in CI by:

1. Installing dependencies: `npm ci`
2. Installing Playwright browsers: `npx playwright install --with-deps`
3. Running the tests: `npm test` 