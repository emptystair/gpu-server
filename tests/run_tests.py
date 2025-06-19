#!/usr/bin/env python3
"""
Test Runner Script

Executes all tests with proper categorization and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Test categories
TEST_CATEGORIES = {
    "unit": "Unit tests (fast, isolated)",
    "integration": "Integration tests (requires services)",
    "performance": "Performance tests (benchmarks)",
    "e2e": "End-to-end tests (full system)",
    "gpu": "GPU-specific tests (requires NVIDIA GPU)",
    "all": "All tests"
}


def run_tests(category: str = "all", verbose: bool = False, coverage: bool = True):
    """Run tests for specified category"""
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add category marker
    if category != "all":
        cmd.extend(["-m", category])
    
    # Add test directory
    cmd.append("tests")
    
    # Additional options
    cmd.extend([
        "--tb=short",
        "--maxfail=10",
        "-x" if category == "unit" else "--continue-on-collection-errors"
    ])
    
    print(f"Running {category} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    return result.returncode


def run_specific_test(test_file: str, test_name: str = None):
    """Run a specific test file or test"""
    cmd = ["pytest", "-vv"]
    
    if test_name:
        cmd.append(f"{test_file}::{test_name}")
    else:
        cmd.append(test_file)
    
    print(f"Running specific test: {test_file}")
    print("-" * 80)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_failed_tests():
    """Re-run only failed tests from last run"""
    cmd = ["pytest", "--lf", "-vv"]
    
    print("Re-running failed tests...")
    print("-" * 80)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def generate_coverage_report():
    """Generate detailed coverage report"""
    print("Generating coverage report...")
    
    # Generate HTML report
    subprocess.run(["coverage", "html"])
    
    # Generate XML report for CI
    subprocess.run(["coverage", "xml"])
    
    # Print report
    subprocess.run(["coverage", "report"])
    
    print(f"\nHTML coverage report available at: htmlcov/index.html")


def main():
    parser = argparse.ArgumentParser(description="GPU OCR Server Test Runner")
    
    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=list(TEST_CATEGORIES.keys()),
        help="Test category to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Run specific test file"
    )
    
    parser.add_argument(
        "-t", "--test",
        help="Run specific test (use with --file)"
    )
    
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Re-run only failed tests"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate coverage report only"
    )
    
    args = parser.parse_args()
    
    # Just generate report
    if args.report:
        generate_coverage_report()
        return 0
    
    # Run failed tests
    if args.failed:
        return run_failed_tests()
    
    # Run specific test
    if args.file:
        return run_specific_test(args.file, args.test)
    
    # Run category
    result = run_tests(
        category=args.category,
        verbose=args.verbose,
        coverage=not args.no_cov
    )
    
    # Generate coverage report if tests passed
    if result == 0 and not args.no_cov:
        print("\n" + "=" * 80)
        generate_coverage_report()
    
    return result


if __name__ == "__main__":
    sys.exit(main())