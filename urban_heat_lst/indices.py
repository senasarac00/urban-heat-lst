"""Radiometric scaling and spectral index calculations (LST in Celsius and NDVI)."""

import numpy as np

# USGS Landsat Collection 2 Level-2 Surface Temperature scaling constants
LST_SCALE_FACTOR = 0.00341802
LST_ADD_OFFSET = 149.0
KELVIN_TO_CELSIUS_OFFSET = 273.15

# USGS Landsat Collection 2 Level-2 Surface Reflectance scaling constants
SR_SCALE_FACTOR = 0.0000275
SR_ADD_OFFSET = -0.2


def compute_lst_celsius(st_b10_dn: np.ndarray) -> np.ndarray:
    """Convert Landsat Collection 2 Level-2 ST_B10 digital numbers to Celsius.

    Formula:
        LST (Kelvin) = DN * 0.00341802 + 149.0
        LST (Celsius) = LST (Kelvin) - 273.15

    Args:
        st_b10_dn: 2D NumPy array of ST_B10 values (with NaN or <=0 for invalid pixels).

    Returns:
        2D float32 NumPy array of Land Surface Temperature in degrees Celsius (°C).
    """
    dn = st_b10_dn.astype(np.float32)
    valid = np.isfinite(dn) & (dn > 0)

    lst_c = np.full(dn.shape, np.nan, dtype=np.float32)
    lst_kelvin = dn[valid] * LST_SCALE_FACTOR + LST_ADD_OFFSET
    lst_c[valid] = lst_kelvin - KELVIN_TO_CELSIUS_OFFSET

    return lst_c


def scale_surface_reflectance(sr_dn: np.ndarray) -> np.ndarray:
    """Apply USGS Landsat Collection 2 Level-2 Surface Reflectance scale and offset.

    Formula:
        SR = DN * 0.0000275 - 0.2

    Args:
        sr_dn: 2D NumPy array of raw Surface Reflectance DN values.

    Returns:
        2D float32 NumPy array of scaled surface reflectance clipped to [0.0, 1.0].
    """
    dn = sr_dn.astype(np.float32)
    valid = np.isfinite(dn) & (dn > 0)

    sr = np.full(dn.shape, np.nan, dtype=np.float32)
    scaled = dn[valid] * SR_SCALE_FACTOR + SR_ADD_OFFSET
    sr[valid] = np.clip(scaled, 0.0001, 1.0)
    return sr


def compute_ndvi(
    sr_b4_dn: np.ndarray,
    sr_b5_dn: np.ndarray,
    already_scaled: bool = False,
) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index (NDVI) from Red and NIR bands.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    Args:
        sr_b4_dn: 2D NumPy array of Landsat 8/9 Red band (SR_B4).
        sr_b5_dn: 2D NumPy array of Landsat 8/9 NIR band (SR_B5).
        already_scaled: Set True if inputs are already scaled reflectance values.

    Returns:
        2D float32 NumPy array of NDVI values in the range [-1.0, 1.0], with NaN for invalid pixels.
    """
    if already_scaled:
        red = sr_b4_dn.astype(np.float32)
        nir = sr_b5_dn.astype(np.float32)
    else:
        red = scale_surface_reflectance(sr_b4_dn)
        nir = scale_surface_reflectance(sr_b5_dn)

    denom = nir + red
    valid = np.isfinite(red) & np.isfinite(nir) & (np.abs(denom) > 1e-6)

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
    ndvi[valid] = np.clip(ndvi[valid], -1.0, 1.0)

    return ndvi
