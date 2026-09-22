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
Tools for tile-based oversampling.
"""

import cv2
import numpy as np
import xarray as xr
from cars.data_structures import cars_dataset


def oversample_depth_map_tile(
    depth_map_tile,
    output_window,
    saving_info,
):
    """
    Oversample one depth map tile to the output window shape.
    """

    out_height = output_window[1] - output_window[0]
    out_width = output_window[3] - output_window[2]

    depth = depth_map_tile["depth"].values.astype(np.float32)
    edges = depth_map_tile["edges"].values.astype(np.float32)
    tile_id = depth_map_tile["tile_id"].values.astype(np.float32)
    normals = depth_map_tile["normals"].values.astype(np.float32)

    depth_out = cv2.resize(depth, (out_width, out_height), cv2.INTER_LINEAR)
    edges_out = cv2.resize(edges, (out_width, out_height), cv2.INTER_NEAREST)
    tile_id_out = cv2.resize(
        tile_id,
        (out_width, out_height),
        cv2.INTER_NEAREST,
    )

    normals_hwc = normals.transpose(1, 2, 0)
    normals_out = cv2.resize(
        normals_hwc,
        (out_width, out_height),
        cv2.INTER_LINEAR,
    ).transpose(2, 0, 1)
    # normalize normals after oversampling
    normals_out /= np.linalg.norm(normals_out, axis=0, keepdims=True)

    out_dataset = xr.Dataset(
        {
            "depth": xr.DataArray(depth_out, dims=("row", "col")),
            "normals": xr.DataArray(normals_out, dims=("axis", "row", "col")),
            "edges": xr.DataArray(
                edges_out.astype(np.int16), dims=("row", "col")
            ),
            "tile_id": xr.DataArray(
                tile_id_out.astype(np.int32),
                dims=("row", "col"),
            ),
        }
    )

    cars_dataset.fill_dataset(
        out_dataset,
        saving_info=saving_info,
        window={
            "row_min": int(output_window[0]),
            "row_max": int(output_window[1]),
            "col_min": int(output_window[2]),
            "col_max": int(output_window[3]),
        },
        profile=None,
        attributes={},
        overlaps={"left": 0, "right": 0, "up": 0, "down": 0},
    )

    return out_dataset
