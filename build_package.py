#!/usr/bin/env python3
"""
Build script for InsideLLM Python SDK package
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"Running: {cmd}")
    if description:
        print(f"Description: {description}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    
    if result.stdout:
        print(result.stdout)
    
    return result

def clean_build():
    """Clean previous build artifacts"""
    print("Cleaning previous build artifacts...")
    
    dirs_to_clean = ['build', 'dist', 'insidellm.egg-info', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}")
    
    # Clean __pycache__ directories recursively
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs[:]:
            if dir_name == '__pycache__':
                shutil.rmtree(os.path.join(root, dir_name))
                dirs.remove(dir_name)
                print(f"Removed {os.path.join(root, dir_name)}")

def install_build_tools():
    """Install required build tools"""
    print("Installing build tools...")
    run_command("pip install --upgrade pip", "Upgrading pip")
    run_command("pip install build twine wheel", "Installing build tools")

def validate_package():
    """Validate package structure"""
    print("Validating package structure...")
    
    required_files = [
        'pyproject.toml',
        'README.md',
        'LICENSE',
        'insidellm/__init__.py',
        'insidellm/client.py',
        'insidellm/models.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        sys.exit(1)
    
    print("Package structure validation passed")

def build_package():
    """Build the package"""
    print("Building the package...")
    run_command("python -m build", "Building wheel and source distribution")

def check_package():
    """Check the built package"""
    print("Checking the built package...")
    run_command("twine check dist/*", "Checking package integrity")

def show_package_info():
    """Show information about the built package"""
    print("\nPackage build completed successfully!")
    print("=" * 50)
    
    # List built files
    if os.path.exists('dist'):
        print("Built files:")
        for file in os.listdir('dist'):
            file_path = os.path.join('dist', file)
            size = os.path.getsize(file_path)
            print(f"  {file} ({size:,} bytes)")
    
    print("\nNext steps:")
    print("1. Test the package locally:")
    print("   pip install dist/insidellm-*.whl")
    
    print("\n2. Upload to Test PyPI (optional):")
    print("   twine upload --repository testpypi dist/*")
    
    print("\n3. Upload to PyPI:")
    print("   twine upload dist/*")
    
    print("\nNote: You'll need PyPI credentials for uploading.")

def main():
    """Main build process"""
    print("InsideLLM Python SDK - Package Builder")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        clean_build()
        install_build_tools()
        validate_package()
        build_package()
        check_package()
        show_package_info()
        
    except KeyboardInterrupt:
        print("\nBuild process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBuild process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()