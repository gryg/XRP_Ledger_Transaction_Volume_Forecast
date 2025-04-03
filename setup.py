from setuptools import setup, find_packages

setup(
    name="xrp-forecasting",
    version="1.0.0",
    description="XRP Transaction Volume Forecasting",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8, <3.12",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.9.0",
        "xgboost>=1.5.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "deepar": ["gluonts>=0.16.0", "torch>=1.10.0"],
        "prophet": ["prophet>=1.1.0"],
        "dev": ["pytest>=6.0.0", "black>=21.5b2", "flake8>=3.9.0"],
    },
)