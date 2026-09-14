#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS Monocular Plugin
# (see https://github.com/CNES/cars-monocular-plugin).
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
this module contains the depth map generation application tools
"""

import numpy as np
from cars.core.cars_logging import logger


def tile_to_tokens(tile_size):
    return (tile_size / 14) ** 2


def tokens_to_tile(tokens):
    return np.floor(np.sqrt(tokens)) * 14


def compute_tile_size_and_overlap(
    mem_constraint,
    moge_model,
    optimal_tile_size=840,
    overlap=28,
    min_tile_size=420,
):
    """
    Compute the best tile size possible given a memory constraint,
    and target optimal / minimum tile sizes for quality.

    """
    models = {
        # empirical approximations
        "vitl-normal": {
            "min_ram": 3430,
            "ram_to_tokens": lambda x: (x - 2352) / 0.3566,
            "tokens_to_ram": lambda x: 0.3566 * x + 2352,
        },
        "vitb-normal": {
            "min_ram": 1600,
            "ram_to_tokens": lambda x: (x - 1360) / 0.3626,
            "tokens_to_ram": lambda x: 0.3626 * x + 1360,
        },
        "vits-normal": {
            "min_ram": 1000,
            "ram_to_tokens": lambda x: (x - 980) / 0.3727,
            "tokens_to_ram": lambda x: 0.3727 * x + 980,
        },
    }

    model = models["vitl-normal"]
    if "vitb-normal" in moge_model:
        model = models["vitb-normal"]
    elif "vits-normal" in moge_model:
        model = models["vits-normal"]
    elif "vitl-normal" not in moge_model:
        logger.warning(
            "The MoGe2 model provided is not recognized. "
            "Memory consumption estimation will be performed "
            "as if vitl-normal was selected."
        )

    optimal_token_count = tile_to_tokens(optimal_tile_size)
    optimal_ram = model["tokens_to_ram"](optimal_token_count)

    if optimal_ram <= mem_constraint or mem_constraint <= model["min_ram"]:

        if mem_constraint <= model["min_ram"]:
            logger.warning(
                "The model selected requires more RAM per worker "
                "than is set as the maximum amount "
                f"(minimum required: {model['min_ram']}MiB, "
                f"maximum per worker: {mem_constraint}MiB)."
            )
            logger.warning(
                "CARS will try to run the model with the optimal tile size."
            )
        # remove overlap from both sides, so that a
        # tile with overlaps everywhere is the biggest one possible
        optimal_window = optimal_tile_size - 2 * overlap
        return optimal_window, overlap

    # highest possible window has to be computed
    min_token_count = tile_to_tokens(min_tile_size)
    best_token_count = model["ram_to_tokens"](mem_constraint)

    if best_token_count < min_token_count:
        logger.warning(
            "The maximum RAM does not allow for a high enough tile size. "
            "The minimum tile size will be used."
        )
        min_window = min_tile_size - 2 * overlap
        return min_window, overlap

    # tile size between optimal and min, that exactly fits the memory provided
    best_tile_size = tokens_to_tile(best_token_count)
    best_window = best_tile_size - 2 * overlap
    return best_window, overlap
