"""
Setup file for the migraine prediction package.
"""
from setuptools import setup, find_packages

setup(
    name="migraine_prediction",
    version="0.1.0",
    description="Migraine prediction model using optimization framework",
    author="MDT Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "joblib>=1.1.0",
        "meta_optimizer>=0.1.0",  # Add dependency on the meta-optimizer package
    ],
    extras_require={
        "explainability": [
            "shap>=0.40.0",
            "lime>=0.2.0",
            "eli5>=0.11.0",
        ],
        "optimization": [
            "deap>=1.3.1",
            "hyperopt>=0.2.7",
            "optuna>=3.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "migraine-predict=migraine_model.cli:main",
        ],
    },
)
