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

import os
from json_checker import Or
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
from json_checker import Checker, OptionalKey

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

        needed_applications = ["depth_map_generation"]

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
        if "save_intermediate_data" not in depth_map_generation_conf:
            depth_map_generation_conf["save_intermediate_data"] = (
                self.save_intermediate_data
            )

        self.depth_map_generation_app = Application(
            "depth_map_generation",
            cfg=depth_map_generation_conf,
        )
        used_conf["depth_map_generation"] = (
            self.depth_map_generation_app.get_conf()
        )

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
        conf["right_image_monocular"] = conf.get(
            "right_image_monocular", False
        )

        conf["activated"] = conf.get("activated", "auto")

        schema = {
            "save_intermediate_data": bool,
            "right_image_monocular": bool,
            "activated": Or(bool, str)
        }

        if conf["activated"] not in (True, False, "auto"):
            raise RuntimeError("The activated parameter should be True, False or auto")

        checker = Checker(schema)
        checker.validate(conf)

        self.save_intermediate_data = conf["save_intermediate_data"]
        self.right_image_monocular = conf["right_image_monocular"]

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
        self.task_progress_id = progress_tree.register_task(
            self.pipeline_progress_id,
            "monocular",
            weight=1.0,
        )
        return self.task_progress_id

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
                self.cars_orchestrator.set_target_task(self.task_progress_id)
                self.depth_map_generation_app.run(
                    image=self.used_conf[INPUT]["sensors"][sensor_key],
                    image_key=sensor_key,
                    dump_folder=depth_map_generation_dump_dir,
                    save_folder=depth_map_generation_save_dir,
                    orchestrator=self.cars_orchestrator,
                )
