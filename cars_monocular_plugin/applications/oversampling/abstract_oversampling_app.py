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
Abstract oversampling application.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.core.cars_logging import logger


@Application.register("oversampling")
class Oversampling(ApplicationTemplate, metaclass=ABCMeta):
    """
    Abstract oversampling application factory.
    """

    available_applications: Dict = {}
    default_application = "simple"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        method = cls.default_application

        if bool(conf) is False or "method" not in conf:
            logger.info(
                "Oversampling method not specified, default %s is used",
                method,
            )
        else:
            method = conf.get("method", cls.default_application)

        if method not in cls.available_applications:
            msg = f"No Oversampling application named {method} registered"
            logger.error(msg)
            raise KeyError(msg)

        logger.info("The Oversampling(%s) application will be used", method)

        return super(Oversampling, cls).__new__(
            cls.available_applications[method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        super().__init__(conf=conf)

    @abstractmethod
    def run(
        self,
        image,
        image_key,
        dump_folder,
        save_folder,
        depth_map_output,
        orchestrator,
        save_intermediate_data=False,
    ):
        """
        Upsample depth_map_generation outputs to original sensor resolution.
        """
