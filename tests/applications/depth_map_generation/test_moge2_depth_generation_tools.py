"""
Test file of depth generation tools
"""

import logging

import pytest

# pylint: disable=line-too-long
from cars_monocular.applications.depth_map_generation.moge2_depth_generation_tools import (  # noqa: E501,B950
    compute_tile_size_and_overlap,
)


@pytest.mark.unit_tests
def test_optimal_tile_fits_memory():
    """
    If optimal RAM fits within the memory constraint,
    the optimal window size should be returned.
    """
    mem_constraint = 10_000  # plenty of RAM
    window, overlap = compute_tile_size_and_overlap(
        mem_constraint=mem_constraint,
        moge_model="vitl-normal",
        optimal_tile_size=840,
        overlap=28,
    )

    assert overlap == 28
    assert window == 840 - 2 * 28


@pytest.mark.unit_tests
def test_memory_below_model_minimum():
    """
    If memory constraint is below the model minimum,
    function still returns optimal tile window.
    """
    mem_constraint = 100
    window, overlap = compute_tile_size_and_overlap(
        mem_constraint=mem_constraint,
        moge_model="vitl-normal",
        optimal_tile_size=840,
        overlap=28,
    )

    assert overlap == 28
    assert window == 840 - 2 * 28


@pytest.mark.unit_tests
def test_unrecognized_model(caplog):
    """
    Unknown model name should trigger a warning and default to vitl-normal.
    """
    caplog.set_level(logging.WARNING)

    mem_constraint = 10_000
    window, overlap = compute_tile_size_and_overlap(
        mem_constraint=mem_constraint,
        moge_model="unknown-model",
        optimal_tile_size=840,
        overlap=28,
    )

    assert overlap == 28
    assert window == 840 - 2 * 28
    assert "not recognized" in caplog.text


@pytest.mark.unit_tests
def test_intermediate_memory_computes_best_tile():
    """
    If memory is between minimum and optimal,
    the function should compute the largest possible tile.
    """
    # intermediate RAM for vitb-normal
    # (optimal window for it is ~550)

    mem_constraint = 2_000

    window, overlap = compute_tile_size_and_overlap(
        mem_constraint=mem_constraint,
        moge_model="vitb-normal",
        optimal_tile_size=840,
        min_tile_size=420,
        overlap=28,
    )

    assert overlap == 28
    # window must be > min window and < optimal window
    assert (420 - 2 * 28) < window < (840 - 2 * 28)
