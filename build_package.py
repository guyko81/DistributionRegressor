#!/usr/bin/env python
"""
Build script for distribution-regressor package.

This script automates the process of:
1. Cleaning old build artifacts
2. Running checks
3. Building the package
4. Validating the built distributions
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    print(f"\n✓ {description} completed successfully!")


def clean_build_artifacts():
    """Remove old build artifacts."""
    print("\n" + "="*70)
    print("Cleaning build artifacts")
    print("="*70)
    
    dirs_to_remove = ['dist', 'build', 'distribution_regressor.egg-info']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            print(f"Removing {dir_name}/")
            shutil.rmtree(dir_name)
    
    print("\n✓ Cleanup completed!")


def check_version_consistency():
    """Check that version is consistent across files."""
    print("\n" + "="*70)
    print("Checking version consistency")
    print("="*70)
    
    # Read version from __init__.py
    init_file = Path('distribution_regressor/__init__.py')
    for line in init_file.read_text().splitlines():
        if line.startswith('__version__'):
            init_version = line.split('"')[1]
            break
    
    # Read version from setup.py
    setup_file = Path('setup.py')
    for line in setup_file.read_text().splitlines():
        if 'version=' in line and not line.strip().startswith('#'):
            setup_version = line.split('"')[1]
            break
    
    # Read version from pyproject.toml
    pyproject_file = Path('pyproject.toml')
    for line in pyproject_file.read_text().splitlines():
        if line.startswith('version ='):
            pyproject_version = line.split('"')[1]
            break
    
    print(f"__init__.py version: {init_version}")
    print(f"setup.py version:    {setup_version}")
    print(f"pyproject.toml version: {pyproject_version}")
    
    if init_version == setup_version == pyproject_version:
        print(f"\n✓ All versions match: {init_version}")
        return init_version
    else:
        print("\n❌ Version mismatch! Please update all version numbers to match.")
        sys.exit(1)


def main():
    """Main build process."""
    print("\n" + "="*70)
    print("Building distribution-regressor package")
    print("="*70)
    
    # Step 1: Clean
    clean_build_artifacts()
    
    # Step 2: Check versions
    version = check_version_consistency()
    
    # Step 3: Build
    run_command(
        "python -m build",
        "Building package"
    )
    
    # Step 4: Validate
    run_command(
        "python -m twine check dist/*",
        "Validating distributions"
    )
    
    # Summary
    print("\n" + "="*70)
    print("Build Summary")
    print("="*70)
    print(f"\n✓ Package built successfully!")
    print(f"  Version: {version}")
    print(f"  Output: dist/")
    
    # List built files
    dist_files = list(Path('dist').glob('*'))
    print(f"\nGenerated files:")
    for f in dist_files:
        size = f.stat().st_size / 1024
        print(f"  - {f.name} ({size:.1f} KB)")
    
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("""
1. Test locally:
   pip install dist/*.whl

2. Upload to TestPyPI (recommended):
   python -m twine upload --repository testpypi dist/*

3. Upload to PyPI:
   python -m twine upload dist/*

See BUILD_AND_PUBLISH.md for detailed instructions.
    """)


if __name__ == "__main__":
    # Check dependencies
    try:
        import build
        import twine
    except ImportError:
        print("\n❌ Missing required packages!")
        print("\nPlease install build tools:")
        print("  pip install --upgrade build twine")
        sys.exit(1)
    
    main()


