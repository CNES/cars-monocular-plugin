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
this module contains the moge2 depth generation application class.
"""

# Standard imports
import os
from pathlib import Path

import cars.orchestrator.orchestrator as ocht
from cars.core.cars_logging import logger
from cars.core.inputs import rasterio_get_size

# CARS imports
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# Third party imports
from json_checker import And, Checker, Or

from .abstract_depth_map_generation_app import DepthMapGeneration
from .moge2_depth_generation_tools import compute_tile_size_and_overlap
from .moge2_wrapper import moge2_wrapper


class MoGe2DepthGeneration(DepthMapGeneration, short_name="moge2"):
    """
    MoGe2DepthGeneration
    """

    def __init__(self, conf=None):
        """
        Init function of MoGe2DepthGeneration

        :param conf: configuration for grid generation
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # Saving files
        self.save_intermediate_data = self.used_config["save_intermediate_data"]
        # check conf
        self.used_method = self.used_config["method"]
        self.model = self.used_config["model"]
        self.edge_threshold = self.used_config["edge_threshold"]

        # Init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # Init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )
        overloaded_conf["method"] = conf.get("method", "moge2")
        overloaded_conf["model"] = conf.get(
            "model", "Ruicheng/moge-2-vitl-normal"
        )
        # replace with actual path to model
        overloaded_conf["model"] = self.check_model_exists(
            overloaded_conf["model"]
        )
        overloaded_conf["edge_threshold"] = conf.get("edge_threshold", 0.6)

        depth_generation_schema = {
            "save_intermediate_data": bool,
            "method": str,
            "model": str,
            "edge_threshold": And(Or(int, float), lambda x: 0 <= x <= 1),
        }

        # Check conf
        checker = Checker(depth_generation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def check_model_exists(self, model):
        """
        Because the model is not packaged with the plugin, it needs to be
        downloaded or imported in some way.
        Check that the model is directly a path, a cached model, or download
        the model.
        Returns the path to the model checkpoint file.
        """
        # if the model is a file
        if os.path.exists(model):
            return model

        # if the model is a recognized hf repo
        models_dir = Path(__file__).parent / "models"
        recognized_repos = {
            "Ruicheng/moge-2-vitl-normal": models_dir / "moge-2-vitl-normal.pt",
            "Ruicheng/moge-2-vitb-normal": models_dir / "moge-2-vitb-normal.pt",
            "Ruicheng/moge-2-vits-normal": models_dir / "moge-2-vits-normal.pt",
        }
        if model in recognized_repos:
            # check if downloaded through cars-download-moge2-model
            if os.path.exists(recognized_repos[model]):
                return recognized_repos[model].absolute().as_posix()

            logger.warning(
                f"The requested MoGe-2 model ({model}) was not downloaded "
                "prior to launching CARS, using cars-download-moge2."
            )

        # if the model is a general hf repo
        try:
            local_model_path = snapshot_download(
                repo_id=model, local_files_only=True
            )

            return os.path.join(local_model_path, "model.pt")

        except LocalEntryNotFoundError:
            logger.warning(
                f"The requested MoGe-2 model ({model}) was not found locally. "
                "CARS will try to download it from HuggingFace."
            )

        try:
            local_model_path = snapshot_download(repo_id=model)

            return os.path.join(local_model_path, "model.pt")
        except Exception as exception:
            logger.error(
                f"The requested MoGe-2 model ({model}) could not be downloaded."
            )
            raise exception

    @cars_profile(name="Depth Map Generation")
    def run(
        self,
        image,
        image_key,
        dump_folder,
        save_folder,
        orchestrator=None,
    ):

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        sensor = image[sens_cst.INPUT_IMG]
        sensor_width, sensor_height = rasterio_get_size(
            sensor["bands"]["b0"]["path"]
        )

        moge_output = cars_dataset.CarsDataset(
            "arrays", name="moge_output_sensor_" + image_key
        )

        window_size, margin = compute_tile_size_and_overlap(
            self.orchestrator.orchestrator_conf["max_ram_per_worker"],
            self.model,
            optimal_tile_size=840,
            overlap=28,
            min_tile_size=420,
        )

        logger.warning(f"Using window size {window_size} with overlap {margin}")

        moge_output.create_grid(
            sensor_width,
            sensor_height,
            window_size,
            window_size,
            margin,
            margin,
        )

        [saving_info] = self.orchestrator.get_saving_infos([moge_output])

        self.orchestrator.add_to_replace_lists(
            moge_output,
            "moge_output",
        )

        if self.save_intermediate_data:
            self.orchestrator.add_to_save_lists(
                os.path.join(dump_folder, "normals.tif"),
                "normals",
                moge_output,
                cars_ds_name="moge_output",
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(dump_folder, "depth.tif"),
                "depth",
                moge_output,
                cars_ds_name="moge_output",
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(dump_folder, "tile_id.tif"),
                "tile_id",
                moge_output,
                cars_ds_name="moge_output",
            )

        self.orchestrator.add_to_save_lists(
            os.path.join(save_folder, "edges.tif"),
            "edges",
            moge_output,
            cars_ds_name="moge_output",
        )

        tile_id = 0
        for row in range(len(moge_output.tiling_grid)):
            for col in range(len(moge_output.tiling_grid[0])):

                window = moge_output.tiling_grid[row, col]
                overlap = moge_output.overlaps[row, col]

                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )

                moge_output[row, col] = self.orchestrator.cluster.create_task(
                    moge2_wrapper
                )(
                    sensor,
                    window,
                    overlap,
                    self.model,
                    self.edge_threshold,
                    full_saving_info,
                    tile_id,
                )

                tile_id += 1

        return moge_output
