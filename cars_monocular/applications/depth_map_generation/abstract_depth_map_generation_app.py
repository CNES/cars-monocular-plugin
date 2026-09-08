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
this module contains the abstract depth map generation application class.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.core.cars_logging import logger


@Application.register("depth_map_generation")
class DepthMapGeneration(ApplicationTemplate, metaclass=ABCMeta):
    """
    AbstractDepthMapGeneration
    """

    available_applications: Dict = {}
    default_application = "moge2"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for depth map generation
        :return: a application_to_use object
        """

        dm_method = cls.default_application

        if bool(conf) is False or "method" not in conf:
            logger.info(
                "Depth map generation method not specified, default "
                " {} is used".format(dm_method)
            )
        else:
            dm_method = conf.get("method", cls.default_application)

        if dm_method not in cls.available_applications:
            logger.error(
                "No DepthMapGeneration application named {} registered".format(
                    dm_method
                )
            )
            raise KeyError(
                "No DepthMapGeneration application named {} registered".format(
                    dm_method
                )
            )

        logger.info(
            "The DepthMapGeneration({}) application will be used".format(
                dm_method
            )
        )

        return super(DepthMapGeneration, cls).__new__(
            cls.available_applications[dm_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of DepthMapGeneration

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(
        self,
        image,
        image_key,
        dump_folder,
        save_folder,
        orchestrator=None,
    ):
        """
        Run DepthMapGeneration application

        Create left (and right if asked for) CarsDataset
        filled with xarray.Dataset, corresponding to left/right :
        - contour mask
        - depth map
        - normal map

        :param image_left: left image
        :type image_left: dict
        :param image_right: right image
        :type image_right: dict
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param orchestrator: orchestrator used
        :param pair_key: pair configuration id
        :type pair_key: str

        :return: left dataset, right dataset
        :rtype: Tuple(CarsDataset, CarsDataset)
        """
