"""
Setup file for the Meta-Optimizer Framework
"""
from setuptools import setup, find_packages

setup(
    name="meta_optimizer_mdt_test",
    version="0.1.1-dev1",
    description="Meta-Optimizer Framework for optimization, meta-learning, explainability, and drift detection",
    author="MDT Team",
    author_email="example@example.com",  # Replace with a valid email
    url="https://github.com/example/meta_optimizer",  # Replace with actual repository URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "joblib>=1.1.0",
        "tqdm>=4.60.0",
        "plotly>=5.5.0",
        "torch>=2.0.0",  # Add PyTorch as a dependency
    ],
    extras_require={
        "explainability": [
            "shap>=0.40.0",
            "lime>=0.2.0",
            "eli5>=0.11.0",
            "interpret>=0.2.7",
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
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    entry_points={
        "console_scripts": [
            "meta-optimizer-mdt-test=meta_optimizer.cli:main",
        ],
    },
    keywords="optimization, meta-learning, explainability, drift-detection, hyperparameter-tuning",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
)
