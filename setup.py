"""
Setup configuration for InsideLLM Python SDK
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="insidellm",
    version="1.0.0",
    author="InsideLLM Team",
    author_email="support@insidellm.com",
    description="Python SDK for LLM/Agent Analytics Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/insidellm/python-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "langchain": ["langchain>=0.1.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "insidellm=insidellm.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)