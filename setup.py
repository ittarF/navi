#!/usr/bin/env python3
"""
Setup script for Navi package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="navi",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A web navigation AI agent that uses local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/navi",
    packages=find_packages(),
    install_requires=[
        "playwright>=1.30.0",
        "httpx>=0.24.1",
        "pillow>=9.5.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "replay": ["opencv-python>=4.7.0.72"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "navi=navi.main:main",
        ],
    },
) 