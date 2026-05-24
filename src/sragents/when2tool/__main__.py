"""
Main entry point for When2Tool pipeline.
Allows running: python -m sragents.when2tool <command> <args>
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from sragents
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if __name__ == "__main__":
    from sragents.when2tool.pipeline import main
    sys.exit(main())
