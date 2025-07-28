from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="unified_ai",
    version="0.1.0",
    author="DRx Mukesh Choudhary",
    author_email="drxmukeshchoudhary@gmail.com",
    description="A powerful and extensible deep learning + machine learning + NLP framework",
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "regex>=2021.8.3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "unified-ai-train=unified_ai.cli.train:main",
            "unified-ai-eval=unified_ai.cli.evaluate:main",
        ],
    },
)