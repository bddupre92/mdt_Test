# End-to-End Testing Suite

This directory contains end-to-end tests for the application, using Playwright.

## Running the Tests

To run the tests, use the following commands:

```bash
# Install dependencies if you haven't yet
npm install

# Run tests
npm test

# Run tests with UI
npm run test:ui
```

## Test Files

- `datasets-page.spec.ts` - Tests for the datasets page functionality including:
  - Tab navigation
  - Dataset generation for different types
  - Dataset exploration and visualization
  - Upload functionality
  - Error handling

## Troubleshooting

If you encounter issues:

1. Make sure the development server is running (`npm run dev`)
2. Ensure all required dependencies are installed
3. Check browser console for errors
4. Verify that the API routes are functioning properly

## Adding New Tests

Follow the pattern in existing tests to add new test cases. Group related tests using `test.describe()` blocks. 