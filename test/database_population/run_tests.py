"""Test runner script for database population tests.

This script provides an easy way to run the database population tests
with various options and configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_pattern=None, verbose=False, coverage=False, stop_on_first=False):
    """Run the database population tests.
    
    Args:
        test_pattern: Optional pattern to filter tests
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        stop_on_first: Stop on first failure
    """
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if stop_on_first:
        cmd.append("-x")
    
    if coverage:
        cmd.extend([
            "--cov=src.database_population",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    if test_pattern:
        cmd.extend(["-k", test_pattern])
    
    # Add common pytest options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Treat unknown markers as errors
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run database population tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                           # Run all tests
  python run_tests.py -v                        # Run with verbose output
  python run_tests.py -k test_batch_processor   # Run only batch processor tests
  python run_tests.py --coverage                # Run with coverage report
  python run_tests.py -x                        # Stop on first failure
  python run_tests.py -k "not integration"      # Skip integration tests
        """
    )
    
    parser.add_argument(
        "-k", "--pattern",
        help="Only run tests matching this pattern"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "-x", "--stop-on-first",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests without running them"
    )
    
    args = parser.parse_args()
    
    if args.list_tests:
        # List available tests
        cmd = [
            "python", "-m", "pytest",
            str(Path(__file__).parent),
            "--collect-only", "-q"
        ]
        subprocess.run(cmd)
        return 0
    
    # Run the tests
    return run_tests(
        test_pattern=args.pattern,
        verbose=args.verbose,
        coverage=args.coverage,
        stop_on_first=args.stop_on_first
    )


if __name__ == "__main__":
    sys.exit(main())