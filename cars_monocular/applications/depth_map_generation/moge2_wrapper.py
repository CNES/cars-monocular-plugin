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
Depth map generation app wrapper functions for CARS Monocular
"""

import warnings

import numpy as np
import torch
import xarray as xr
from cars.core.inputs import rasterio_get_size, rasterio_read_as_array
from cars.data_structures import cars_dataset
from moge.model.v2 import MoGeModel
from rasterio.windows import Window


def moge2_wrapper(
    sensor, window, overlap, model_name, edge_threshold, saving_info, tile_id
):
    """
    The main wrapper for the MoGe2 depth generation application.
    """

    max_img_size = rasterio_get_size(sensor["bands"]["b0"]["path"])
    overlap, token_count = add_insufficient_overlap(
        window, overlap, max_img_size
    )

    # load image
    input_image = get_sensor_data(sensor, window, overlap)

    # init moge2
    device = torch.device("cpu")
    with warnings.catch_warnings():  # suppress prints from pytorch
        warnings.simplefilter("ignore")
        model = MoGeModel.from_pretrained(model_name).to(device)

    input_image = torch.tensor(input_image, dtype=torch.float32, device=device)

    # run model
    with warnings.catch_warnings():  # suppress prints from pytorch
        warnings.simplefilter("ignore")
        output = model.infer(
            input_image, use_fp16=False, num_tokens=token_count
        )

    # format data
    out_dataset = format_moge_output(
        output, overlap, tile_id, edge_threshold=edge_threshold
    )

    cars_dataset.fill_dataset(
        out_dataset,
        saving_info=saving_info,
        window={
            "row_min": window[0],
            "row_max": window[1],
            "col_min": window[2],
            "col_max": window[3],
        },
        profile=None,
        attributes={},
        # we override overlaps and trim our data after running the model
        overlaps={"left": 0, "right": 0, "up": 0, "down": 0},
    )

    # cleanup
    del model
    del input_image
    del output

    return out_dataset


def add_insufficient_overlap(
    window, overlap, max_img_size, token_window_size=14
):
    """
    From the window, minimum overlap and max image size, compute the best
    new overlap to fit an exact amount of tokens for the tile.
    Returns the new overlap, and the optimal token count.
    """

    # Current height/width including overlap
    base_height = window[1] + overlap[1] - window[0] + overlap[0]
    base_width = window[3] + overlap[3] - window[2] + overlap[2]

    # How much extra we need to reach
    insufficient_height = (
        token_window_size - (base_height % token_window_size)
    ) % token_window_size
    insufficient_width = (
        token_window_size - (base_width % token_window_size)
    ) % token_window_size

    # Try to add the missing height WITHOUT exceeding the full image boundaries
    top_space = max(window[0] - overlap[0], 0)
    bottom_space = max_img_size[1] - window[1] - overlap[1]

    add_top = min(insufficient_height // 2, top_space)
    add_bottom = insufficient_height - add_top
    add_bottom = min(add_bottom, bottom_space)
    add_top = min(insufficient_height - add_bottom, top_space)

    overlap[0] += add_top
    overlap[1] += add_bottom

    # Try to add the missing width WITHOUT exceeding the full image boundaries
    left_space = max(window[2] - overlap[2], 0)
    right_space = max_img_size[0] - window[3] - overlap[3]

    add_left = min(insufficient_width // 2, left_space)
    add_right = insufficient_width - add_left
    add_right = min(add_right, right_space)
    add_left = min(insufficient_width - add_right, left_space)

    overlap[2] += add_left
    overlap[3] += add_right

    # Final dimensions after adjustment
    final_height = window[1] + overlap[1] - window[0] + overlap[0]
    final_width = window[3] + overlap[3] - window[2] + overlap[2]

    token_count = np.ceil(final_height * final_width / token_window_size**2)

    return overlap, token_count


def get_edges(normals, threshold=0.6):
    vertical = np.zeros_like(normals).astype(np.float32)
    vertical[..., -1] = 1
    dot_product = np.sum(np.abs(normals) * vertical, axis=-1)
    return (dot_product < threshold).astype(np.int16)


def format_moge_output(output, overlap, tile_id, edge_threshold=0.6):
    """
    Format the output given by MoGe into an xr.DataArray contaning :
    - depth (row, col)
    - normals (axis, row, col)
    - edges (row, col)
    - tile_id (row, col)
    """

    def to_numpy(arr):
        return arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr

    def remove_overlap(arr):
        top, bottom, left, right = overlap
        return arr[top : (-bottom or None), left : (-right or None), ...]

    data_vars = {}

    depth = to_numpy(output["depth"])
    normals = to_numpy(output["normal"])
    edges = get_edges(normals, threshold=edge_threshold)

    depth = remove_overlap(depth)
    normals = remove_overlap(normals)
    edges = remove_overlap(edges)
    tile_id_arr = np.zeros_like(edges, dtype=np.int32) + tile_id

    normals = normals.transpose(2, 0, 1)

    data_vars["depth"] = xr.DataArray(depth, dims=("row", "col"))
    data_vars["normals"] = xr.DataArray(normals, dims=("axis", "row", "col"))
    data_vars["edges"] = xr.DataArray(edges, dims=("row", "col"))
    data_vars["tile_id"] = xr.DataArray(tile_id_arr, dims=("row", "col"))

    return xr.Dataset(data_vars)


def normalize_band(band, percentiles=(2, 98)):
    """
    Normalize the given band using percentiles as bounds.
    Returns the normalized data in the interval [0, 1]
    """
    p_low, p_high = np.percentile(band, percentiles)
    if p_high == p_low:
        return np.zeros_like(band, dtype=np.float32)
    band_n = np.clip((band - p_low) / (p_high - p_low), 0, 1)
    return band_n


def get_sensor_data(sensor, window, overlap):
    """
    Returns the sensor image data in a window with the overlap taken into
    account. If the image doesn't have 3+ bands, create them:
    - If 1 band -> replicate to 3 (gray -> 'RGB' gray).
    - If 2 bands -> stack band0, band1, and their average.
    - If 3 bands -> keep as is.
    - If >3 bands -> keep first 3.
    """
    nb_bands = len(sensor["bands"])

    def get_band(sensor, band):
        band_i, _ = rasterio_read_as_array(
            sensor["bands"][band]["path"],
            Window.from_slices(
                (
                    window[0] - overlap[0],
                    window[1] + overlap[1],
                ),
                (
                    window[2] - overlap[2],
                    window[3] + overlap[3],
                ),
            ),
        )
        if band_i.ndim > 2:
            band_i = band_i[sensor["bands"][band]["band"]]
        band_i = normalize_band(band_i)
        return band_i

    if nb_bands == 1:  # RGB = (b0, b0, b0)
        b_0 = get_band(sensor, "b0")

        data = np.repeat(b_0[None, :, :], 3, axis=0)

    elif nb_bands == 2:  # RGB = (b0, b1, (b0+b1)/2)
        b_0 = get_band(sensor, "b0")
        b_1 = get_band(sensor, "b1")

        data = np.stack([b_0, b_1, (b_0 + b_1) / 2], axis=0)

    else:  # RGB = (b0, b1, b2)
        b_0 = get_band(sensor, "b0")
        b_1 = get_band(sensor, "b1")
        b_2 = get_band(sensor, "b2")

        data = np.stack([b_0, b_1, b_2], axis=0)

    return data


def ensure_three_channels(img: np.ndarray) -> np.ndarray:
    """
    Convert any image (H x W), (H x W x C), or multi-band image
    into a normalized 3-channel uint8 image using robust statistics.

    Rules:
    - If 1 band -> replicate to 3 (gray -> RGB).
    - If 2 bands -> stack band0, band1, and their average (recommended).
    - If 3 bands -> keep as is.
    - If >3 bands -> keep first 3.
    """

    def norm_band(band):
        p_low, p_high = np.percentile(band, (2, 98))
        if p_high == p_low:
            return np.zeros_like(band, dtype=np.uint8)
        band_n = np.clip((band - p_low) / (p_high - p_low), 0, 1)
        return (band_n * 255).astype(np.uint8)

    if img.ndim == 2:
        img = img[None, :, :]  # add band dimension

    channels, _, _ = img.shape

    # --- Handle number of bands ---
    if channels == 1:
        # 1 -> replicate to RGB
        out = np.repeat(norm_band(img[0, :, :])[None, :, :], 3, axis=0)

    elif channels == 2:
        # 2 ->
        #   - band0 -> R
        #   - band1 -> G
        #   - average -> B
        band_0 = norm_band(img[0, :, :])
        band_1 = norm_band(img[1, :, :])
        avg = norm_band((img[0, :, :] + img[1, :, :]) / 2)
        out = np.stack([band_0, band_1, avg], axis=2)

    elif channels >= 3:
        # 3 -> Use the first 3 bands
        out = np.stack(
            [
                norm_band(img[0, :, :]),
                norm_band(img[1, :, :]),
                norm_band(img[2, :, :]),
            ],
            axis=2,
        )

    return out
