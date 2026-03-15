"""METRICX: Quantitative Evaluation Framework for XAI Methods."""

from setuptools import setup, find_packages

setup(
    name="metricx",
    version="0.1.0",
    description=(
        "METRICX: Quantitative Evaluation Framework for Post-hoc "
        "Interpretability Methods"
    ),
    author="UFMG DCC – Master's Dissertation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.42.0",
        "lime>=0.2.0.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
