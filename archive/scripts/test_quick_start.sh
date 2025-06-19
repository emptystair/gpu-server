#!/bin/bash

# Quick start script for testing PDFs with GPU OCR Server

echo "=== GPU OCR Server - Quick Test Start ==="
echo
echo "This script will:"
echo "1. Build the Docker image with all dependencies"
echo "2. Run verification to check setup"
echo "3. Process 5 PDFs as a test"
echo
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Run Docker-based tests
./run_docker_tests.sh --build --test-type verify

echo
echo "Verification complete. Now running OCR tests on 5 PDFs..."
echo "Press Enter to continue..."
read

./run_docker_tests.sh --test-type direct --max-files 5

echo
echo "=== Test Complete ==="
echo "Check the results in:"
echo "  - test-results/test_results_comprehensive.json"
echo
echo "To run more comprehensive tests, use:"
echo "  ./run_docker_tests.sh --test-type all"