#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS Monocular
# (see https://github.com/CNES/cars-monocular).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Init file of CARS Monocular
"""

import cars_monocular.applications  # noqa: F401
import cars_monocular.pipelines  # noqa: F401
from cars_monocular.download_models import ensure_default_model_available

# Best-effort auto-download of the default model on import.
ensure_default_model_available()
