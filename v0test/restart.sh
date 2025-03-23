#!/bin/bash
# Stop any running Next.js processes
pkill -f "next dev" || true
# Start the dev server
cd $(dirname "$0")
npm run dev 