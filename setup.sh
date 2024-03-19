#!/bin/bash

# Activate conda environment with Python 3.11.6
conda activate python3.11.6

poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Command to run all pre-commit hooks
poetry run pre-commit run --hook-stage push
