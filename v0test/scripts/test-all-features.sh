#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  MIGRAINE DT COMPREHENSIVE TESTING  ${NC}"
echo -e "${BLUE}=====================================${NC}"

# Make sure we're in the project directory
cd "$(dirname "$0")/.." || exit 1

# First check if the server is running
echo -e "\n${YELLOW}Checking if the development server is running...${NC}"
if ! curl -s http://localhost:3000 > /dev/null; then
  echo -e "${RED}Development server is not running!${NC}"
  echo -e "${YELLOW}Starting development server...${NC}"
  npm run dev &
  SERVER_PID=$!
  echo -e "${GREEN}Development server started with PID $SERVER_PID${NC}"
  echo "Waiting for server to be ready..."
  sleep 10
else
  echo -e "${GREEN}Development server is already running.${NC}"
  SERVER_PID=""
fi

# Run tests
echo -e "\n${YELLOW}Running Playwright tests...${NC}"
npm test

# Get test result
TEST_RESULT=$?

# Generate a report summary
echo -e "\n${YELLOW}Test Report Summary:${NC}"
if [ $TEST_RESULT -eq 0 ]; then
  echo -e "${GREEN}✓ All tests passed successfully!${NC}"
else
  echo -e "${RED}✗ Some tests failed. Check the report for details.${NC}"
  echo -e "${YELLOW}Opening test report...${NC}"
  npx playwright show-report
fi

# Clean up - kill the server if we started it
if [ -n "$SERVER_PID" ]; then
  echo -e "\n${YELLOW}Stopping development server...${NC}"
  kill $SERVER_PID
  echo -e "${GREEN}Development server stopped.${NC}"
fi

echo -e "\n${BLUE}=====================================${NC}"
echo -e "${BLUE}        TESTING COMPLETED            ${NC}"
echo -e "${BLUE}=====================================${NC}"

exit $TEST_RESULT 