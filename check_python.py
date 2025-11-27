#!/usr/bin/env python3
"""Check if Python 3.12 is being used."""

import sys

def check_python_version():
    """Check if Python version is 3.12.x."""
    version = sys.version_info
    if version.major == 3 and version.minor == 12:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        print("✓ Python version is correct for this project")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} detected")
        print("✗ This project requires Python 3.12")
        print("\nTo install Python 3.12:")
        print("  brew install python@3.12")
        print("  # or")
        print("  pyenv install 3.12.9")
        print("\nThen configure Poetry:")
        print("  poetry env use python3.12")
        return False

if __name__ == "__main__":
    success = check_python_version()
    sys.exit(0 if success else 1)

