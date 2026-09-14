# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CODIP
#
#     https://gitlab.cnes.fr/co3d-image/codip

# GLOBAL VARIABLES
# Set Virtualenv directory name
.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

ifndef VENV
	VENV := venv
endif

CHECK_CARS = $(shell ${VENV}/bin/python -m pip list|grep "cars ")
CHECK_CARS_LIBGEO = $(shell ${CARS_VENV}/bin/python -m pip list|grep cars-monocular-plugin)


# Source and tests directory
DIR = "cars_monocular_plugin"
VERSION := $(shell /bin/sh .version.sh)
ifeq ($(LOCAL_WHL_DIR),)
export LOCAL_WHL_DIR=/usr/local/share/codip
endif


# Python global variables definition
PYTHON_VERSION_MIN = 3.9

PYTHON=$(shell command -v python3)

PYTHON_VERSION_CUR=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')

############### Check python version supported ############

ifeq (, $(PYTHON))
    $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif



################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: git
git: ## init local git repository if not present
	@test -d .git/ || git init .


.PHONY: venv
venv:
	@test -d $(VENV) || { \
		python3 -m venv $(VENV); \
		echo "build env"; \
		echo $(VENV); \
	}
	@$(VENV)/bin/python3 -m pip install --upgrade pip setuptools wheel setuptools-scm

.PHONY: install
install: venv  ## install plugin
	@[ "${CHECK_CARS}" ] || test -f ${VENV}/bin/cars_monocular_plugin || ${VENV}/bin/pip install -e .[cars,dev]


.PHONY: install-dev
install-dev: venv git  ## install the package in dev mode in virtualenv
	@[ "${CHECK_CARS}" ] || test -f ${VENV}/bin/cars_monocular_plugin || ${VENV}/bin/pip install --no-binary rasterio,fiona -e .[dev,cars]
	@[ -z "${CHECK_CARS}" ] || test -f ${VENV}/bin/cars_monocular_plugin || ${VENV}/bin/pip install --no-binary rasterio,fiona -e .[dev]
	@test -f .git/hooks/pre-commit || echo "Install pre-commit"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@echo " cars_monocular_plugin venv usage : source ${VENV}/bin/activates"


.PHONY: test
test: install-dev ## run all tests + coverage html
	@${VENV}/bin/pytest -o log_cli=true -o log_cli_level=${LOGLEVEL} --cov-config=.coveragerc --cov-report html --cov


## Code quality, linting section

### Format with isort and black

.PHONY: format
format: install format/isort format/black  ## run black and isort formatting (depends install)

.PHONY: format/isort
format/isort: install  ## run isort formatting (depends install)
	@echo "+ $@"
	@${VENV}/bin/isort cars_monocular_plugin tests

.PHONY: format/black
format/black: install  ## run black formatting (depends install)
	@echo "+ $@"
	@${VENV}/bin/black cars_monocular_plugin tests

### Check code quality and linting : isort, black, flake8, pylint

.PHONY: lint
lint:  lint/isort lint/black lint/flake8 lint/pylint ## check code quality and linting (source venv before)

.PHONY: lint/isort
lint/isort: ## check imports style with isort
	@echo "+ $@"
	@${VENV}/bin/isort --check cars_monocular_plugin  tests
	
.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${VENV}/bin/black --check cars_monocular_plugin  tests

.PHONY: lint/flake8
lint/flake8: ## check linting with flake8
	@echo "+ $@"
	@${VENV}/bin/flake8 cars_monocular_plugin  tests

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@set -o pipefail; ${VENV}/bin/pylint cars_monocular_plugin tests --rcfile=.pylintrc --output-format=parseable | tee pylint-report.txt # pipefail to propagate pylint exit code in bash

.PHONY: clean
clean: ## clean: remove venv, cars build, cache, ...
	@rm -rf ${VENV}
	@rm -rf dist
	@rm -rf build
	@rm -rf **/__pycache__
	@rm -f .coverage
	@rm -rf .coverage.*

.PHONY: wheel
wheel: ## build *.whl package and put it on local registry
	@rm -f dist/*.whl
	@export LIBGEO_PLUGIN_VERSION=$(VERSION) && python3 setup.py bdist_wheel
ifeq ($(JENKINS_CI),True)
	@echo "On Jenkins CI the package is published to artifactory registry"
else
	@cp dist/*.whl $(LOCAL_WHL_DIR)
endif
