#!/usr/bin/env python
# coding: utf8

# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CODIP
#
#     https://gitlab.cnes.fr/co3d-image/codip
"""
Setup.py for CARS Monocular
"""

from setuptools import setup

try:
    setup(use_scm_version={"fallback_version": "0.0.0"})
except Exception:
    print(
        "\n\nAn error occurred while building the project, "
        "please ensure you have the most updated version of pip, setuptools, "
        "setuptools_scm and wheel with:\n"
        "   pip install -U pip setuptools setuptools_scm wheel\n\n"
    )
    raise
