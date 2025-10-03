#!/usr/bin/env python3
"""Quick launcher for the Memoirr chat interface.

This script provides a convenient way to launch the Gradio chat interface
from the project root directory.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from src.frontend.gradio_app import main
    main()