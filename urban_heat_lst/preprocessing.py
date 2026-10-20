"""Cloud, shadow, snow, and water masking with QA_PIXEL and AOI raster clipping."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Sequence

import geopandas as gpd
import numpy as np
import pystac
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

logger = logging.getLogger(__name__)

# Landsat Collection 2 Level-2 QA_PIXEL bit positions
BIT_FILL = 0
BIT_DILATED_CLOUD = 1
BIT_CIRRUS = 2
BIT_CLOUD = 3
BIT_CLOUD_SHADOW = 4
BIT_SNOW = 5
BIT_CLEAR = 6
BIT_WATER = 7

DEFAULT_QA_MASK_BITS = (
    BIT_FILL,
    BIT_DILATED_CLOUD,
    BIT_CIRRUS,
    BIT_CLOUD,
    BIT_CLOUD_SHADOW,
    BIT_SNOW,
    BIT_WATER,
)


@dataclass
class ClippedBands:
    """Container for clipped and masked raw Landsat band arrays and metadata."""

    st_b10: np.ndarray
    sr_b4: np.ndarray
    sr_b5: np.ndarray
    qa_pixel: np.ndarray
    valid_mask: np.ndarray
    profile: Dict[str, Any]


def build_qa_mask(
    qa_array: np.ndarray,
    mask_bits: Sequence[int] = DEFAULT_QA_MASK_BITS,
) -> np.ndarray:
    """Build a boolean mask of valid (clear) pixels from Landsat C2 QA_PIXEL band.

    Landsat 8/9 Collection 2 QA_PIXEL bit definitions:
      - Bit 0: Fill
      - Bit 1: Dilated Cloud
      - Bit 2: Cirrus
      - Bit 3: Cloud
      - Bit 4: Cloud Shadow
      - Bit 5: Snow
      - Bit 6: Clear
      - Bit 7: Water

    Args:
        qa_array: 2D integer NumPy array of QA_PIXEL values.
        mask_bits: Sequence of bit positions (0-15) to exclude when set to 1.
            Defaults to [0, 1, 2, 3, 4, 5, 7] (Fill, Dilated Cloud, Cirrus,
            Cloud, Cloud Shadow, Snow, and Water).

    Returns:
        2D boolean NumPy array where True indicates a valid pixel (none of mask_bits are set).
    """
    qa_int = qa_array.astype(np.uint16)

    invalid_bitmask = 0
    for bit in mask_bits:
        invalid_bitmask |= 1 << int(bit)

    is_valid = (qa_int & invalid_bitmask) == 0
    return is_valid


def _resolve_asset_href(item: pystac.Item, candidates: list[str]) -> str:
    """Find the signed HREF for a band asset given a list of candidate keys."""
    for key in candidates:
        if key in item.assets:
            return item.assets[key].href
    raise KeyError(
        f"None of the expected band keys {candidates} found in STAC item assets: {list(item.assets.keys())}"
    )


def _read_and_clip_band(
    href: str, aoi_gdf: gpd.GeoDataFrame
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Read a Cloud-Optimized GeoTIFF window clipped to the AOI geometry.

    Args:
        href: URL or file path to the raster asset.
        aoi_gdf: GeoDataFrame representing the Area of Interest.

    Returns:
        Tuple of (2D clipped NumPy array, updated rasterio profile).
    """
    with rasterio.open(href) as src:
        aoi_projected = aoi_gdf.to_crs(src.crs)
        geoms = [mapping(geom) for geom in aoi_projected.geometry]
        out_image, out_transform = rio_mask(src, geoms, crop=True, filled=True, nodata=0)
        profile = src.profile.copy()
        profile.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        return out_image[0], profile


def load_and_preprocess_scene(
    item: pystac.Item,
    aoi_gdf: gpd.GeoDataFrame,
    qa_mask_bits: Sequence[int] = DEFAULT_QA_MASK_BITS,
) -> ClippedBands:
    """Clip required Landsat bands to the AOI and apply the QA_PIXEL bitmask.

    Supports both Planetary Computer common band names ('lwir11', 'red', 'nir08', 'qa_pixel')
    and standard USGS band names ('ST_B10', 'SR_B4', 'SR_B5', 'QA_PIXEL').

    Args:
        item: Signed STAC Item from Microsoft Planetary Computer.
        aoi_gdf: GeoDataFrame of the Area of Interest.
        qa_mask_bits: Sequence of QA_PIXEL bit indices to mask out (default: [0, 1, 2, 3, 4, 5, 7]).

    Returns:
        ClippedBands dataclass containing masked float arrays and GeoTIFF profile.
    """
    href_b10 = _resolve_asset_href(item, ["lwir11", "ST_B10"])
    href_b4 = _resolve_asset_href(item, ["red", "SR_B4"])
    href_b5 = _resolve_asset_href(item, ["nir08", "SR_B5"])
    href_qa = _resolve_asset_href(item, ["qa_pixel", "QA_PIXEL"])

    logger.info(
        "Clipping QA_PIXEL, ST_B10, SR_B4 (Red), and SR_B5 (NIR) to AOI (masking QA bits %s)...",
        list(qa_mask_bits),
    )
    qa_raw, profile = _read_and_clip_band(href_qa, aoi_gdf)
    b10_raw, _ = _read_and_clip_band(href_b10, aoi_gdf)
    b4_raw, _ = _read_and_clip_band(href_b4, aoi_gdf)
    b5_raw, _ = _read_and_clip_band(href_b5, aoi_gdf)

    # Ensure valid mask excludes outside-AOI nodata (0) and any pixel matching qa_mask_bits
    qa_valid = build_qa_mask(qa_raw, mask_bits=qa_mask_bits)
    valid_mask = qa_valid & (b10_raw > 0) & (b4_raw > 0) & (b5_raw > 0)

    valid_pct = float(np.count_nonzero(valid_mask)) / float(valid_mask.size) * 100.0
    logger.info("Valid clear pixels after QA_PIXEL masking: %.2f%%", valid_pct)

    st_b10 = np.where(valid_mask, b10_raw.astype(np.float32), np.nan)
    sr_b4 = np.where(valid_mask, b4_raw.astype(np.float32), np.nan)
    sr_b5 = np.where(valid_mask, b5_raw.astype(np.float32), np.nan)

    return ClippedBands(
        st_b10=st_b10,
        sr_b4=sr_b4,
        sr_b5=sr_b5,
        qa_pixel=qa_raw,
        valid_mask=valid_mask,
        profile=profile,
    )
