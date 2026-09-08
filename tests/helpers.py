"""
Depth map generation tests helper functions
"""

import json
import logging

# Standard imports
import os
import tempfile
from shutil import copy2

import numpy as np
import rasterio as rio

# CARS imports
from cars.pipelines.parameters import sensor_inputs
from scipy.ndimage import zoom


def cars_copy2(in_data, out_data):
    """
    Copy raster file from in_data to out_data,
    adding overviews to out_data
    """

    with rio.open(in_data, "r+") as src:
        data = src.read()
        nodata = src.nodata

        nb_bands = data.shape[0]
        if nb_bands == 1:
            selected_data = np.concatenate([data, data, data], axis=0)
        elif nb_bands == 2:
            selected_data = np.concatenate([data, data[1:2]], axis=0)
        else:
            selected_data = data[:3]

        # Create mask for valid data
        if nodata is not None:
            valid_mask = selected_data != nodata
        else:
            valid_mask = ~np.isnan(selected_data)

        # Only process valid data for normalization
        valid_data = selected_data[valid_mask]

        if len(valid_data) > 0:
            # Normalize only valid data to uint8 range
            data_min = np.nanmin(valid_data)
            data_max = np.nanmax(valid_data)

            if data_max > data_min:
                data_uint8 = np.full_like(selected_data, 0, dtype="uint8")
                # Normalize valid data
                normalized = (
                    (selected_data - data_min) / (data_max - data_min) * 254
                ).astype("uint8")
                data_uint8[valid_mask] = normalized[valid_mask]
                # Set nodata pixels to 255 (valid range for uint8)
                data_uint8[~valid_mask] = 255
            else:
                # All valid pixels have the same value
                data_uint8 = np.full_like(selected_data, 127, dtype="uint8")
                data_uint8[~valid_mask] = 255
        else:
            # No valid data
            data_uint8 = np.full_like(selected_data, 255, dtype="uint8")

    # downsample for overview
    zoom_factor = (1, 0.25, 0.25)
    data_uint8 = zoom(data_uint8, zoom_factor, order=1)

    # Save as PNG overview
    base_name = os.path.splitext(out_data)[0]
    png_path = f"{base_name}_overview.png"

    png_profile = {
        "driver": "PNG",
        "height": src.height,
        "width": src.width,
        "count": data_uint8.shape[0],
        "dtype": "uint8",
    }

    with rio.open(png_path, "w", **png_profile) as png_dst:
        png_dst.write(data_uint8)

    # Copy file
    copy2(in_data, out_data)


def temporary_dir():
    """
    Returns path to temporary dir from CARS_TEST_TEMPORARY_DIR environment
    variable. Defaults to default temporary directory
    (/tmp or TMPDIR environment variable)
    """
    if "CARS_TEST_TEMPORARY_DIR" not in os.environ:
        # return default tmp dir
        logging.info(
            "CARS_TEST_TEMPORARY_DIR is not set, "
            "cars will use default temporary directory instead"
        )
        return tempfile.gettempdir()
    # return env defined tmp dir
    return os.environ["CARS_TEST_TEMPORARY_DIR"]


def assert_same_images(actual, expected, rtol=0, atol=0):
    """
    Compare two image files with assertion:
    * same height, width, transform, crs
    * assert_allclose() on numpy buffers
    """
    with rio.open(actual) as rio_actual:
        with rio.open(expected) as rio_expected:
            np.testing.assert_equal(rio_actual.width, rio_expected.width)
            np.testing.assert_equal(rio_actual.height, rio_expected.height)
            assert rio_actual.transform == rio_expected.transform
            assert rio_actual.crs == rio_expected.crs
            assert rio_actual.nodata == rio_expected.nodata
            data1 = rio_actual.read()
            data2 = rio_expected.read()
            data1[data1 == rio_actual.nodata] = 0
            data1[np.isnan(data1)] = 0
            data2[data2 == rio_expected.nodata] = 0
            data2[np.isnan(data2)] = 0
            np.testing.assert_allclose(data1, data2, rtol=rtol, atol=atol)


def generate_input_json(
    input_json,
    output_directory,
    orchestrator_mode,
    orchestrator_parameters=None,
):
    """
    Load a partially filled input.json, fill it with output directory
    and orchestrator mode, and transform relative path to
     absolute paths. Generates a new json dumped in output directory

    :param input_json: input json
    :type input_json: str
    :param output_directory: absolute path out directory
    :type output_directory: str
    :param orchestrator_mode: orchestrator mode
    :type orchestrator_mode: str
    :param orchestrator_parameters: advanced orchestrator params
    :type orchestrator_parameters: dict

    :return: path of generated json, dict input config
    :rtype: str, dict
    """
    # Load dict
    json_dir_path = os.path.dirname(input_json)
    with open(input_json, "r", encoding="utf8") as fstream:
        config = json.load(fstream)
    # Overload orchestrator
    config["orchestrator"] = {"mode": orchestrator_mode}
    if orchestrator_mode in ("mp", "multiprocessing"):
        config["orchestrator"]["per_job_timeout"] = 120
    if orchestrator_parameters is not None:
        config["orchestrator"].update(orchestrator_parameters)
    # Overload output directory
    config["output"] = {"directory": os.path.join(output_directory, "output")}

    config["advanced"] = {
        "pipeline": "monocular",
    }

    # Create keys
    if "monocular" not in config:
        config["monocular"] = {"applications": {}, "advanced": {}}

    # transform paths
    new_config = config.copy()

    new_config["input"] = sensor_inputs.sensors_check_inputs(
        new_config["input"], config_dir=json_dir_path
    )

    # dump json
    new_json_path = os.path.join(output_directory, "new_input.json")
    with open(new_json_path, "w", encoding="utf8") as fstream:
        json.dump(new_config, fstream, indent=2)

    return new_json_path, new_config


def absolute_data_path(data_path):
    """
    Return a full absolute path to test data
    environment variable.
    """
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_folder, data_path)
