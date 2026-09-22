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
Simple oversampling implementation.
"""

import cars.orchestrator.orchestrator as ocht
import numpy as np
from cars.core.cars_logging import logger
from cars.core.inputs import rasterio_get_size
from cars.data_structures import cars_dataset
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from json_checker import Checker

from .abstract_oversampling_app import Oversampling
from .simple_oversampling_tools import oversample_depth_map_tile


class SimpleOversampling(Oversampling, short_name="simple"):
    """
    Very simple raster oversampling to original sensor size.
    """

    def __init__(self, conf=None):
        super().__init__(conf=conf)
        self.used_method = self.used_config["method"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

    def check_conf(self, conf):
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        overloaded_conf["method"] = conf.get("method", "simple")
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        checker = Checker({"method": str, "save_intermediate_data": bool})
        checker.validate(overloaded_conf)
        return overloaded_conf

    @staticmethod
    def _compute_scaled_tiling_grid(
        source_tiling_grid,
        source_width,
        source_height,
        target_width,
        target_height,
    ):
        """
        Scale a full tiling grid from source resolution to target resolution.
        """

        scaled_grid = np.zeros_like(source_tiling_grid)

        for row in range(source_tiling_grid.shape[0]):
            for col in range(source_tiling_grid.shape[1]):
                row_min, row_max, col_min, col_max = source_tiling_grid[
                    row, col
                ]

                scaled_row_min = int(
                    round(row_min * target_height / source_height)
                )
                scaled_row_max = int(
                    round(row_max * target_height / source_height)
                )
                scaled_col_min = int(
                    round(col_min * target_width / source_width)
                )
                scaled_col_max = int(
                    round(col_max * target_width / source_width)
                )

                scaled_grid[row, col, 0] = max(
                    0, min(target_height, scaled_row_min)
                )
                scaled_grid[row, col, 1] = max(
                    0, min(target_height, scaled_row_max)
                )
                scaled_grid[row, col, 2] = max(
                    0, min(target_width, scaled_col_min)
                )
                scaled_grid[row, col, 3] = max(
                    0, min(target_width, scaled_col_max)
                )

        return scaled_grid

    def run(
        self,
        image,
        image_key,
        dump_folder,
        save_folder,
        depth_map_output,
        orchestrator,
    ):
        sensor = image[sens_cst.INPUT_IMG]
        target_width, target_height = rasterio_get_size(
            sensor["bands"]["b0"]["path"]
        )

        source_tiling_grid = depth_map_output.tiling_grid
        source_width = int(np.max(source_tiling_grid[:, :, 3]))
        source_height = int(np.max(source_tiling_grid[:, :, 1]))

        scaled_tiling_grid = self._compute_scaled_tiling_grid(
            source_tiling_grid,
            source_width,
            source_height,
            target_width,
            target_height,
        )

        oversampling_output = cars_dataset.CarsDataset(
            "arrays", name="oversampling_output_sensor_" + image_key
        )
        oversampling_output.tiling_grid = scaled_tiling_grid
        oversampling_output.overlaps = np.zeros_like(scaled_tiling_grid)
        oversampling_output.generate_none_tiles()

        [saving_info] = orchestrator.get_saving_infos([oversampling_output])

        orchestrator.add_to_replace_lists(
            oversampling_output,
            "oversampling_output",
        )

        if self.save_intermediate_data:
            orchestrator.add_to_save_lists(
                dump_folder + "/normals.tif",
                "normals",
                oversampling_output,
                cars_ds_name="oversampling_output",
            )
            orchestrator.add_to_save_lists(
                dump_folder + "/depth.tif",
                "depth",
                oversampling_output,
                cars_ds_name="oversampling_output",
            )
            orchestrator.add_to_save_lists(
                dump_folder + "/tile_id.tif",
                "tile_id",
                oversampling_output,
                cars_ds_name="oversampling_output",
            )

        orchestrator.add_to_save_lists(
            save_folder + "/edges.tif",
            "edges",
            oversampling_output,
            cars_ds_name="oversampling_output",
        )

        for row in range(depth_map_output.tiling_grid.shape[0]):
            for col in range(depth_map_output.tiling_grid.shape[1]):
                full_saving_info = ocht.update_saving_infos(
                    saving_info,
                    row=row,
                    col=col,
                )

                oversampling_output[
                    row, col
                ] = orchestrator.cluster.create_task(oversample_depth_map_tile)(
                    depth_map_output[row, col],
                    scaled_tiling_grid[row, col],
                    full_saving_info,
                )

        logger.debug(
            "Oversampling scheduled for sensor %s from %sx%s to %sx%s",
            image_key,
            source_width,
            source_height,
            target_width,
            target_height,
        )

        return oversampling_output
