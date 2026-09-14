"""
Test file of depth generation moge2 wrapper functions
"""

import numpy as np
import pytest

# pylint: disable=line-too-long
from cars_monocular_plugin.applications.depth_map_generation.moge2_wrapper import (  # noqa: E501,B950
    add_insufficient_overlap,
)


def compute_dims(window, overlap):
    width = window[1] + overlap[1] - window[0] + overlap[0]
    height = window[3] + overlap[3] - window[2] + overlap[2]
    return width, height


@pytest.mark.unit_tests
def test_no_insufficient_overlap_needed():
    """
    If base dimensions are already divisible by token_window_size,
    overlap should not change.
    """
    window = [10, 150, 20, 160]
    overlap = [28, 28, 28, 28]
    max_img_size = (300, 300)
    token_window_size = 14

    base_w, base_h = compute_dims(window, overlap)
    assert base_w % token_window_size == 0
    assert base_h % token_window_size == 0

    overlap_copy = overlap.copy()
    new_overlap, token_count = add_insufficient_overlap(
        window,
        overlap,
        max_img_size,
        token_window_size,
    )

    assert new_overlap == overlap_copy
    assert token_count == np.ceil(base_h * base_w / token_window_size**2)


@pytest.mark.unit_tests
def test_add_overlap_symmetrically_when_space_allows():
    """
    Missing pixels should be split between both sides when space allows.
    """
    window = [50, 160, 40, 150]
    overlap = [5, 5, 5, 5]
    max_img_size = (300, 300)
    token_window_size = 14

    base_h, base_w = compute_dims(window, overlap)

    new_overlap, token_count = add_insufficient_overlap(
        window,
        overlap,
        max_img_size,
        token_window_size,
    )

    final_h, final_w = compute_dims(window, new_overlap)

    assert final_h % token_window_size == 0
    assert final_w % token_window_size == 0
    assert final_h >= base_h
    assert final_w >= base_w
    assert token_count == np.ceil(final_h * final_w / token_window_size**2)


@pytest.mark.unit_tests
def test_overlap_limited_by_image_boundaries_height():
    """
    If there is not enough space on one side (top),
    additional overlap must be added on the other side (bottom).
    """
    window = [20, 120, 50, 78]
    overlap = [19, 10, 0, 0]
    max_img_size = (140, 200)
    token_window_size = 14

    new_overlap, _ = add_insufficient_overlap(
        window,
        overlap,
        max_img_size,
        token_window_size,
    )

    final_h, _ = compute_dims(window, new_overlap)

    assert final_h % token_window_size == 0
    assert new_overlap[0] <= window[0]  # top overlap limited
    assert new_overlap[1] >= overlap[1]  # bottom absorbs remainder


@pytest.mark.unit_tests
def test_overlap_limited_by_image_boundaries_width():
    """
    If there is not enough space on one side (right),
    additional overlap must be added on the other side (left).
    """
    window = [20, 140, 90, 198]
    overlap = [5, 5, 4, 2]
    max_img_size = (200, 200)
    token_window_size = 14

    _, base_w = compute_dims(window, overlap)

    new_overlap, _ = add_insufficient_overlap(
        window,
        overlap,
        max_img_size,
        token_window_size,
    )

    _, final_w = compute_dims(window, new_overlap)

    assert final_w % token_window_size == 0
    assert new_overlap[3] <= max_img_size[1] - window[3]
    assert final_w >= base_w


@pytest.mark.unit_tests
def test_token_count_rounds_up():
    """
    Token count must always be ceil(height * width / token_window_size²).
    """
    window = [0, 101, 0, 101]
    overlap = [0, 0, 0, 0]
    max_img_size = (200, 200)
    token_window_size = 14

    new_overlap, token_count = add_insufficient_overlap(
        window,
        overlap,
        max_img_size,
        token_window_size,
    )

    final_h, final_w = compute_dims(window, new_overlap)

    expected = np.ceil(final_h * final_w / token_window_size**2)

    assert token_count == expected
    assert new_overlap == [0, 11, 0, 11]
