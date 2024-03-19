#!/bin/bash
# Add your code for the job manager

poetry run python -m project.main --config-name=cifar10-sgd

poetry run python -m project.main --config-name=cifar10-adam

poetry run python -m project.main --config-name=cifar10-dgnmc

poetry run python -m project.main --config-name=cifar10-dgnexact

poetry run python -m project.main --config-name=cifar10-bdgnmc

poetry run python -m project.main --config-name=cifar10-bdgnexact

poetry run python -m project.main --config-name=cifar10-lbfgs