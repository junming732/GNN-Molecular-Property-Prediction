from setuptools import setup, find_packages

setup(
    name="gnn-molecular-property-prediction",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "ase>=3.22.0",
        "optuna>=2.10.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)