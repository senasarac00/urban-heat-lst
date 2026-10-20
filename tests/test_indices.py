"""Unit tests for LST Celsius conversion and NDVI calculation."""

import numpy as np

from urban_heat_lst.indices import (
    KELVIN_TO_CELSIUS_OFFSET,
    LST_ADD_OFFSET,
    LST_SCALE_FACTOR,
    compute_lst_celsius,
    compute_ndvi,
)


def test_compute_lst_celsius_scaling() -> None:
    """Verify exact Landsat C2 L2 ST_B10 scaling to Celsius and NaN handling."""
    # Pick DN = 45000 -> Kelvin = 45000 * 0.00341802 + 149.0 = 302.8109 K -> 29.6609 °C
    dn_array = np.array([[45000.0, 0.0], [50000.0, np.nan]], dtype=np.float32)
    lst_c = compute_lst_celsius(dn_array)

    expected_00 = 45000.0 * LST_SCALE_FACTOR + LST_ADD_OFFSET - KELVIN_TO_CELSIUS_OFFSET
    expected_10 = 50000.0 * LST_SCALE_FACTOR + LST_ADD_OFFSET - KELVIN_TO_CELSIUS_OFFSET

    assert np.isclose(lst_c[0, 0], expected_00, atol=1e-3)
    assert np.isclose(lst_c[1, 0], expected_10, atol=1e-3)
    assert np.isnan(lst_c[0, 1])
    assert np.isnan(lst_c[1, 1])


def test_compute_ndvi_values_and_range() -> None:
    """Verify NDVI computation for vegetation vs bare soil and invalid pixels."""
    # High NIR, Low Red -> positive NDVI
    sr_red = np.array([[0.05, 0.25], [0.20, np.nan]], dtype=np.float32)
    sr_nir = np.array([[0.45, 0.25], [0.10, 0.30]], dtype=np.float32)

    ndvi = compute_ndvi(sr_red, sr_nir, already_scaled=True)

    assert np.isclose(ndvi[0, 0], (0.45 - 0.05) / (0.45 + 0.05), atol=1e-5)
    assert np.isclose(ndvi[0, 1], 0.0, atol=1e-5)
    assert np.isclose(ndvi[1, 0], (0.10 - 0.20) / (0.10 + 0.20), atol=1e-5)
    assert np.isnan(ndvi[1, 1])
