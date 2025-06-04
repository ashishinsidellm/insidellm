# InsideLLM SDK - Packaging and Distribution Guide

This guide explains how to build, test, and distribute the InsideLLM Python SDK as a pip package.

## Prerequisites

1. Python 3.8+ installed
2. Git repository set up
3. PyPI account (for distribution)
4. TestPyPI account (for testing, optional)

## Project Structure

```
insidellm-python-sdk/
├── insidellm/                 # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── client.py             # Main client
│   ├── models.py             # Event models
│   ├── config.py             # Configuration
│   ├── queue_manager.py      # Async processing
│   ├── langchain_integration.py
│   ├── decorators.py         # Function decorators
│   ├── context_manager.py    # Context managers
│   ├── exceptions.py         # Custom exceptions
│   ├── utils.py              # Utilities
│   └── py.typed              # Type hint marker
├── examples/                 # Usage examples
│   ├── basic_usage.py
│   ├── langchain_example.py
│   └── custom_agent_example.py
├── pyproject.toml           # Modern Python packaging config
├── setup.py                 # Legacy setup (optional)
├── README.md                # Package documentation
├── LICENSE                  # MIT license
├── MANIFEST.in              # Additional files to include
├── build_package.py         # Build automation script
└── PACKAGING_GUIDE.md       # This guide
```

## Build Configuration Files

### pyproject.toml
Modern Python packaging configuration with:
- Project metadata
- Dependencies
- Build system requirements
- Optional dependencies (langchain, dev tools)
- Tool configurations (black, mypy, pytest)

### MANIFEST.in
Specifies additional files to include in the distribution:
- Documentation files
- License
- Examples
- Excludes test files and development artifacts

## Building the Package

### Method 1: Using the Build Script (Recommended)

```bash
# Make the script executable
chmod +x build_package.py

# Run the build script
python build_package.py
```

The script will:
1. Clean previous build artifacts
2. Install required build tools
3. Validate package structure
4. Build wheel and source distributions
5. Check package integrity
6. Show next steps

### Method 2: Manual Build Process

1. **Install build tools:**
```bash
pip install --upgrade pip build twine wheel
```

2. **Clean previous builds:**
```bash
rm -rf build/ dist/ *.egg-info/
```

3. **Build the package:**
```bash
python -m build
```

4. **Check the package:**
```bash
twine check dist/*
```

## Testing the Package Locally

### Install from local wheel:
```bash
pip install dist/insidellm-*.whl
```

### Install in development mode:
```bash
pip install -e .
```

### Install with optional dependencies:
```bash
pip install -e .[langchain,dev]
```

### Test the installation:
```python
import insidellm
print(insidellm.__version__)

# Test basic functionality
insidellm.initialize(api_key="test-key")
client = insidellm.get_client()
print("SDK imported and initialized successfully")
```

## Distribution Options

### 1. Upload to Test PyPI (Recommended for testing)

```bash
# Configure Test PyPI credentials
pip install keyring
keyring set https://test.pypi.org/legacy/ __token__

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI to test
pip install --index-url https://test.pypi.org/simple/ insidellm
```

### 2. Upload to PyPI (Production)

```bash
# Configure PyPI credentials
keyring set https://upload.pypi.org/legacy/ __token__

# Upload to PyPI
twine upload dist/*

# Install from PyPI
pip install insidellm
```

### 3. GitHub Releases

1. Create a GitHub release with version tag (e.g., `v1.0.0`)
2. Upload the built wheel and source distribution files
3. Users can install directly from GitHub:

```bash
pip install https://github.com/insidellm/python-sdk/archive/v1.0.0.tar.gz
```

## Version Management

### Update version in pyproject.toml:
```toml
[project]
version = "1.0.1"
```

### Automated versioning (optional):
Use tools like `bump2version` or `semantic-release` for automated version management.

## Continuous Integration

### GitHub Actions example (.github/workflows/build.yml):

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: pytest
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*

  publish:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: "3.11"
    
    - name: Build package
      run: |
        python -m pip install --upgrade pip build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Package Quality Checklist

### Before publishing:
- [ ] All tests pass
- [ ] Documentation is complete
- [ ] Version number is updated
- [ ] CHANGELOG.md is updated
- [ ] License is included
- [ ] Package builds without errors
- [ ] Package installs correctly
- [ ] Examples work with the installed package
- [ ] Dependencies are pinned appropriately
- [ ] Security vulnerabilities checked

### Tools for quality assurance:
```bash
# Code formatting
black insidellm/

# Linting
flake8 insidellm/

# Type checking
mypy insidellm/

# Security scanning
bandit -r insidellm/

# Dependency checking
safety check
```

## Troubleshooting

### Common issues:

1. **Import errors after installation:**
   - Check package structure
   - Verify __init__.py files exist
   - Ensure dependencies are installed

2. **Build failures:**
   - Check pyproject.toml syntax
   - Verify all required files exist
   - Clean build directory and retry

3. **Upload failures:**
   - Check PyPI credentials
   - Verify package name is available
   - Ensure version number is unique

4. **Installation issues:**
   - Check Python version compatibility
   - Verify dependency versions
   - Use virtual environment for testing

### Getting help:
- Python Packaging Guide: https://packaging.python.org/
- PyPI Help: https://pypi.org/help/
- Setuptools Documentation: https://setuptools.pypa.io/

## Best Practices

1. **Use semantic versioning** (MAJOR.MINOR.PATCH)
2. **Test on multiple Python versions**
3. **Keep dependencies minimal and well-pinned**
4. **Provide comprehensive documentation**
5. **Include usage examples**
6. **Use continuous integration**
7. **Monitor package downloads and issues**
8. **Respond to user feedback promptly**

## Maintenance

### Regular tasks:
- Update dependencies
- Fix security vulnerabilities
- Add new features based on user feedback
- Update documentation
- Monitor performance and usage metrics

### Release process:
1. Update version number
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test package locally
5. Upload to Test PyPI and test
6. Create GitHub release
7. Upload to PyPI
8. Announce release