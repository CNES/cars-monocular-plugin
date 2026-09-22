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
# pylint: disable=too-many-lines
# attribute-defined-outside-init is disabled so that we can create and use
# attributes however we need, to stick to the "everything is attribute" logic
# introduced in issue#895
# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-nested-blocks
"""
CARS Monocular pipeline class file
"""

# Standard imports
from __future__ import print_function

import copy
import json
import os

import yaml
from cars.applications.application import Application
from cars.core.cars_logging import logger

# CARS imports
from cars.core.progress.progress import ProgressTree
from cars.core.utils import safe_makedirs
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters, sensor_inputs
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUT,
    ORCHESTRATOR,
    OUTPUT,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.subsampling.subsampling import SubsamplingPipeline
from json_checker import And, Checker, OptionalKey, Or

package_path = os.path.dirname(__file__)

PIPELINE = "pipeline"
MONOCULAR = "monocular"


@Pipeline.register(MONOCULAR)
class Monocular(PipelineTemplate):
    """
    Monocular pipeline
    """

    def __init__(self, conf, config_dir=None):
        """
        Instantiates Monocular pipeline

        :param conf: user conf as a dict
        :param config_dir: configuration directory
        """
        self.used_conf = self.check_conf(conf, config_dir)

        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        self.dump_dir = os.path.join(self.out_dir, "dump_dir")

    def check_conf(self, conf, config_dir):
        """
        Check the configuration and returns the used conf, with all
        default values in the correct position
        """
        config_dir = os.path.abspath(config_dir) if config_dir else None

        self.check_global_schema(conf)

        used_conf = {}
        used_conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )
        used_conf[INPUT] = self.check_inputs(conf, config_dir)
        used_conf[MONOCULAR] = conf.get(MONOCULAR, {})

        user_app = conf.get(MONOCULAR, {}).get(APPLICATIONS, {})
        user_adv = conf.get(MONOCULAR, {}).get(ADVANCED, {})

        used_conf[MONOCULAR][ADVANCED] = self.check_advanced(user_adv)
        used_conf[MONOCULAR][APPLICATIONS] = self.check_applications(user_app)
        used_conf[OUTPUT] = self.check_output(conf)

        return used_conf

    def check_global_schema(self, conf):

        # at least input and output are needed
        global_schema = {
            INPUT: dict,
            OUTPUT: dict,
            OptionalKey(MONOCULAR): dict,
            OptionalKey(PIPELINE): str,
            OptionalKey(ADVANCED): dict,
            OptionalKey(ORCHESTRATOR): dict,
        }

        checker_inputs = Checker(global_schema)
        checker_inputs.validate(conf)

    def check_inputs(self, conf, config_dir=None):
        """
        Check the inputs given to the pipeline. They can only be sensor images.
        """
        return sensor_inputs.sensors_check_inputs(
            conf.get(INPUT, {}), config_dir=config_dir
        )

    def check_applications(self, conf):
        """
        Check the application configurations given to the pipeline
        """

        used_conf = {}

        needed_applications = ["depth_map_generation", "oversampling"]

        # Check if all specified applications are used
        # Application in terrain_application are note used in
        # the sensors_to_dense_depth_maps pipeline
        for app_key in conf.keys():
            if app_key not in needed_applications:
                msg = (
                    f"No {app_key} application used in the "
                    + "default Cars pipeline"
                )
                logger.error(msg)
                raise NameError(msg)

        depth_map_generation_conf = conf.get("depth_map_generation", {})
        oversampling_conf = conf.get("oversampling", {})
        if (
            "save_intermediate_data" not in depth_map_generation_conf
            or depth_map_generation_conf["save_intermediate_data"] is False
        ):
            depth_map_generation_conf["save_intermediate_data"] = (
                self.save_intermediate_data
            )
        if (
            "save_intermediate_data" not in oversampling_conf
            or oversampling_conf["save_intermediate_data"] is False
        ):
            oversampling_conf["save_intermediate_data"] = (
                self.save_intermediate_data
            )

        self.depth_map_generation_app = Application(
            "depth_map_generation",
            cfg=depth_map_generation_conf,
        )
        self.oversampling_app = Application(
            "oversampling", cfg=oversampling_conf
        )

        used_conf["depth_map_generation"] = (
            self.depth_map_generation_app.get_conf()
        )
        used_conf["oversampling"] = self.oversampling_app.get_conf()

        return used_conf

    def check_output(self, conf):
        """
        Check the output parameters given to the pipeline.
        """
        return output_parameters.check_output_parameters(
            conf[INPUT], conf[OUTPUT], 1
        )[0]

    def check_advanced(self, conf):
        """
        Check the advanced parameters for the pipeline
        """
        conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )
        conf["right_image_monocular"] = conf.get("right_image_monocular", False)

        conf["activated"] = conf.get("activated", "auto")
        conf["resolution"] = conf.get("resolution", 1)

        schema = {
            "save_intermediate_data": bool,
            "right_image_monocular": bool,
            "resolution": And(int, lambda x: x > 0),
            "activated": Or(bool, str),
        }

        if conf["activated"] not in (True, False, "auto"):
            raise RuntimeError(
                "The activated parameter should be True, False or auto"
            )

        checker = Checker(schema)
        checker.validate(conf)

        self.save_intermediate_data = conf["save_intermediate_data"]
        self.right_image_monocular = conf["right_image_monocular"]
        self.resolution = conf["resolution"]
        self.use_subsampling = self.resolution > 1

        return conf

    def setup_progress_tracking(self, parent_pipeline_id=None):
        """
        Setup progress tracking for monocular.

        :param parent_pipeline_id: Optional parent pipeline ID
        :type parent_pipeline_id: int or None
        :return: Task ID for the monocular task
        :rtype: int
        """
        progress_tree = ProgressTree()
        if parent_pipeline_id is None:
            self.pipeline_progress_id = progress_tree.begin_pipeline(
                "Monocular"
            )
        else:
            self.pipeline_progress_id = parent_pipeline_id

        self.subsampling_pipeline_progress_id = None
        if self.use_subsampling:
            self.subsampling_pipeline_progress_id = (
                progress_tree.begin_pipeline(
                    "Subsampling",
                    parent_id=self.pipeline_progress_id,
                )
            )

        self.depth_map_task_progress_id = progress_tree.register_task(
            self.pipeline_progress_id,
            "depth_map_generation",
            weight=20.0,
        )

        return self.depth_map_task_progress_id

    @staticmethod
    def load_subsampling_inputs(out_dir, resolution):
        """
        Load subsampling-generated inputs for a given resolution.
        """

        yaml_file = os.path.join(
            out_dir,
            "subsampling",
            "res_" + str(resolution),
            "input.yaml",
        )

        with open(yaml_file, encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return json.loads(json.dumps(data, indent=4))

    @cars_profile(name="Run_monocular", interval=0.5)
    def run(
        self, args=None, parent_pipeline_id=None
    ):  # pylint: disable=unused-argument
        """
        Exécute le pipeline Monocular

        :param args: parsed command-line arguments
        :param parent_pipeline_id: Optional pipeline ID for progress tracking
        """
        self.setup_progress_tracking(parent_pipeline_id)

        sensors_to_compute = [
            left for left, _ in self.used_conf[INPUT]["pairing"]
        ]
        if self.right_image_monocular:
            sensors_to_compute += [
                right for _, right in self.used_conf[INPUT]["pairing"]
            ]
        sensors_to_compute = set(sensors_to_compute)

        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            out_yaml_path=os.path.join(
                self.out_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as self.cars_orchestrator:

            inputs_for_depth_map = self.used_conf[INPUT]

            # ---- Subsampling ----
            if self.use_subsampling:
                logger.info(
                    "Starting image subsampling before depth map generation"
                )
                subsampling_conf = {
                    INPUT: copy.deepcopy(self.used_conf[INPUT]),
                    OUTPUT: {out_cst.OUT_DIRECTORY: self.dump_dir},
                    "subsampling": {
                        ADVANCED: {"resolutions": [self.resolution]}
                    },
                }
                subsampling_pipeline = SubsamplingPipeline(subsampling_conf)
                subsampling_pipeline.run(
                    parent_pipeline_id=self.subsampling_pipeline_progress_id
                )

                inputs_for_depth_map = self.load_subsampling_inputs(
                    self.dump_dir,
                    self.resolution,
                )

            # --- Depth map generation ---
            for sensor_key in sensors_to_compute:

                depth_map_generation_save_dir = os.path.join(
                    self.out_dir, "monocular", sensor_key
                )
                depth_map_generation_dump_dir = os.path.join(
                    self.dump_dir, "depth_map_generation", sensor_key
                )
                safe_makedirs(depth_map_generation_save_dir)
                safe_makedirs(depth_map_generation_dump_dir)

                logger.info(
                    f"Starting Depth map generation for sensor {sensor_key}"
                )
                self.cars_orchestrator.set_target_task(
                    self.depth_map_task_progress_id
                )
                depth_map_output = self.depth_map_generation_app.run(
                    image=inputs_for_depth_map["sensors"][sensor_key],
                    image_key=sensor_key,
                    dump_folder=depth_map_generation_dump_dir,
                    save_folder=depth_map_generation_save_dir,
                    orchestrator=self.cars_orchestrator,
                )

                if self.use_subsampling:

                    oversampling_save_dir = os.path.join(
                        self.out_dir, "monocular", sensor_key
                    )
                    oversampling_dump_dir = os.path.join(
                        self.dump_dir, "oversampling", sensor_key
                    )
                    safe_makedirs(oversampling_save_dir)
                    safe_makedirs(oversampling_dump_dir)

                    self.oversampling_app.run(
                        image=self.used_conf[INPUT]["sensors"][sensor_key],
                        image_key=sensor_key,
                        dump_folder=oversampling_dump_dir,
                        save_folder=oversampling_save_dir,
                        depth_map_output=depth_map_output,
                        orchestrator=self.cars_orchestrator,
                    )
