"""
Setup file for the package.
"""
from setuptools import setup, find_packages

setup(
    name="mdt_test",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "sqlalchemy>=1.4.0",
        "alembic>=1.7.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-multipart>=0.0.6",
        "pydantic[email]>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "httpx>=0.24.0",
    ],
)
