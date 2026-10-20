"""Unit tests for QA_PIXEL cloud/shadow/snow/water bitmasking and seasonal STAC selection."""

from datetime import datetime, timezone
import numpy as np
import pystac

from urban_heat_lst.preprocessing import (
    BIT_CIRRUS,
    BIT_CLEAR,
    BIT_CLOUD,
    BIT_CLOUD_SHADOW,
    BIT_DILATED_CLOUD,
    BIT_FILL,
    BIT_SNOW,
    BIT_WATER,
    build_qa_mask,
)
from urban_heat_lst.stac_search import select_least_cloudy_item


def test_build_qa_mask_with_configurable_bit_list() -> None:
    """Verify QA_PIXEL bitmasking using customizable bit lists (including Bit 5 Snow and Bit 7 Water)."""
    clear_land = 1 << BIT_CLEAR
    cloud_pixel = (1 << BIT_CLOUD) | (1 << BIT_DILATED_CLOUD)
    shadow_pixel = 1 << BIT_CLOUD_SHADOW
    snow_pixel = 1 << BIT_SNOW
    fill_pixel = 1 << BIT_FILL
    water_pixel = (1 << BIT_CLEAR) | (1 << BIT_WATER)
    cirrus_pixel = 1 << BIT_CIRRUS

    qa = np.array(
        [
            [clear_land, cloud_pixel, shadow_pixel, snow_pixel],
            [cirrus_pixel, fill_pixel, water_pixel, clear_land],
        ],
        dtype=np.uint16,
    )

    # Default bits: [0, 1, 2, 3, 4, 5, 7] -> masks fill, dilated cloud, cirrus, cloud, shadow, snow (5), water (7)
    mask_default = build_qa_mask(qa, mask_bits=[0, 1, 2, 3, 4, 5, 7])
    expected_default = np.array(
        [
            [True, False, False, False],
            [False, False, False, True],
        ]
    )
    np.testing.assert_array_equal(mask_default, expected_default)

    # Without Bit 7 (water) in mask_bits -> water pixel remains True, snow (Bit 5) is still False
    mask_keep_water = build_qa_mask(qa, mask_bits=[0, 1, 2, 3, 4, 5])
    expected_keep_water = np.array(
        [
            [True, False, False, False],
            [False, False, True, True],
        ]
    )
    np.testing.assert_array_equal(mask_keep_water, expected_keep_water)


def test_select_least_cloudy_item_with_season_filter() -> None:
    """Verify that select_least_cloudy_item filters out non-summer scenes and picks lowest cloud cover."""
    geom = {
        "type": "Polygon",
        "coordinates": [[[30.5, 36.8], [30.9, 36.8], [30.9, 37.1], [30.5, 37.1], [30.5, 36.8]]],
    }
    # Winter scene with 0.0% cloud cover (should be excluded by default June-August filter [6, 7, 8])
    item_winter = pystac.Item(
        id="scene_winter_zero_cloud",
        geometry=geom,
        bbox=[30.5, 36.8, 30.9, 37.1],
        datetime=datetime(2024, 1, 15, 8, 30, tzinfo=timezone.utc),
        properties={"eo:cloud_cover": 0.0},
    )
    item_summer_cloudy = pystac.Item(
        id="scene_summer_cloudy",
        geometry=geom,
        bbox=[30.5, 36.8, 30.9, 37.1],
        datetime=datetime(2024, 7, 1, 8, 30, tzinfo=timezone.utc),
        properties={"eo:cloud_cover": 12.4},
    )
    item_summer_clear = pystac.Item(
        id="scene_summer_clear",
        geometry=geom,
        bbox=[30.5, 36.8, 30.9, 37.1],
        datetime=datetime(2024, 8, 15, 8, 30, tzinfo=timezone.utc),
        properties={"eo:cloud_cover": 0.8},
    )

    selected = select_least_cloudy_item(
        [item_winter, item_summer_cloudy, item_summer_clear],
        aoi_geojson=geom,
        season_months=[6, 7, 8],
    )
    assert selected.id == "scene_summer_clear"
