"""Urban Heat Island (UHI) classification, vector hotspot extraction, and statistics."""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes, sieve
from scipy.stats import pearsonr
from shapely.geometry import shape

logger = logging.getLogger(__name__)

CLASS_LABELS: Dict[int, str] = {
    1: "Very Low (< Mean - 1.5 SD)",
    2: "Low (Mean - 1.5 SD to Mean - 0.5 SD)",
    3: "Moderate (Mean - 0.5 SD to Mean + 0.5 SD)",
    4: "High (Mean + 0.5 SD to Mean + 1.5 SD)",
    5: "Very High / Hotspot (>= Mean + 1.5 SD)",
}


@dataclass
class UHIStatistics:
    """Summary statistics of Land Surface Temperature, NDVI, and UHI classes."""

    scene_id: str
    acquisition_date: str
    min_lst_c: float
    max_lst_c: float
    mean_lst_c: float
    std_lst_c: float
    median_lst_c: float
    mean_ndvi: float
    lst_ndvi_corr: float
    lst_ndvi_p_value: float
    valid_pixel_count: int
    hotspot_area_ha: float


def classify_lst(
    lst_c: np.ndarray,
    valid_mask: np.ndarray | None = None,
    worldcover: np.ndarray | None = None,
    urban_only: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Classify Land Surface Temperature into 5 UHI classes based on Mean ± SD of valid pixels.

    Classes (uint8, 0 = NoData):
      - 1 (Very Low):  LST < μ - 1.5 * σ
      - 2 (Low):       μ - 1.5 * σ <= LST < μ - 0.5 * σ
      - 3 (Moderate):  μ - 0.5 * σ <= LST < μ + 0.5 * σ
      - 4 (High):      μ + 0.5 * σ <= LST < μ + 1.5 * σ
      - 5 (Very High): LST >= μ + 1.5 * σ

    Args:
        lst_c: 2D float NumPy array of LST in Celsius (NaN for NoData).
        valid_mask: Optional boolean mask of valid pixels after QA_PIXEL masking.
        worldcover: Optional 2D uint8 array of aligned ESA WorldCover classes.
        urban_only: If True, compute μ and σ strictly from WorldCover Built-up (50) pixels
            and apply classification only to Built-up pixels.

    Returns:
        Tuple of (2D uint8 classified array, list of 4 threshold values [t1, t2, t3, t4]).

    Raises:
        ValueError: If the LST array contains no valid finite values.
    """
    valid = np.isfinite(lst_c)
    if valid_mask is not None:
        valid = valid & valid_mask.astype(bool)

    if urban_only:
        if worldcover is None:
            raise ValueError("worldcover array is required when urban_only=True.")
        valid = valid & (worldcover == 50)

    if not np.any(valid):
        raise ValueError("LST array contains no valid finite pixels to classify.")

    valid_pixels = lst_c[valid]
    mean_val = float(np.mean(valid_pixels))
    std_val = float(np.std(valid_pixels))

    thresholds = [
        mean_val - 1.5 * std_val,
        mean_val - 0.5 * std_val,
        mean_val + 0.5 * std_val,
        mean_val + 1.5 * std_val,
    ]

    classified = np.zeros(lst_c.shape, dtype=np.uint8)
    # np.digitize returns 0..4 for 4 thresholds; add 1 to map to classes 1..5
    classified[valid] = np.digitize(valid_pixels, bins=thresholds).astype(np.uint8) + 1

    return classified, thresholds


def _compute_min_pixels_for_area(profile: Dict[str, Any], min_area_ha: float) -> int:
    """Calculate the minimum pixel count threshold corresponding to min_area_ha."""
    if min_area_ha <= 0.0:
        return 1
    transform = profile["transform"]
    pixel_area_m2 = abs(float(transform.a) * float(transform.e))
    crs = profile.get("crs")
    if crs is not None and crs.is_geographic:
        # Approximate degree to meter conversion if CRS is geographic
        pixel_area_m2 *= 111_320.0 * 111_320.0
    if pixel_area_m2 <= 0.0:
        return 1
    min_area_m2 = min_area_ha * 10_000.0
    return max(1, int(np.ceil(min_area_m2 / pixel_area_m2)))


def extract_hotspot_vectors(
    classified_lst: np.ndarray,
    profile: Dict[str, Any],
    min_class: int = 5,
    min_area_ha: float = 0.5,
) -> gpd.GeoDataFrame:
    """Filter small patches with rasterio.features.sieve and polygonize hotspots.

    Args:
        classified_lst: 2D uint8 array of UHI classes (1..5, 0 = NoData).
        profile: Rasterio profile containing 'transform' and 'crs'.
        min_class: Minimum class value to extract as hotspot (default: 5).
        min_area_ha: Minimum contiguous hotspot area in hectares (default: 0.5 ha).

    Returns:
        GeoDataFrame containing hotspot polygons, class ID, class label, and area in m² / ha.
    """
    hotspot_binary = np.where(classified_lst >= min_class, classified_lst, 0).astype(
        np.uint8
    )
    min_pixels = _compute_min_pixels_for_area(profile, min_area_ha)

    if np.any(hotspot_binary > 0) and min_pixels > 1:
        logger.info(
            "Applying rasterio.features.sieve with size=%d pixels (min_area_ha=%.2f ha)...",
            min_pixels,
            min_area_ha,
        )
        sieved_hotspots = sieve(hotspot_binary, size=min_pixels, connectivity=8)
    else:
        sieved_hotspots = hotspot_binary

    hotspot_mask = sieved_hotspots >= min_class
    transform = profile["transform"]
    crs = profile.get("crs")

    records = []
    if np.any(hotspot_mask):
        for geom_dict, value in shapes(
            sieved_hotspots, mask=hotspot_mask, transform=transform
        ):
            class_id = int(value)
            poly = shape(geom_dict)
            if poly.is_empty:
                continue
            records.append(
                {
                    "class_id": class_id,
                    "class_label": CLASS_LABELS.get(class_id, f"Class {class_id}"),
                    "geometry": poly,
                }
            )

    if not records:
        gdf = gpd.GeoDataFrame(
            columns=["class_id", "class_label", "area_m2", "area_ha", "geometry"],
            geometry="geometry",
            crs=crs,
        )
        return gdf

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    # Calculate metric area (if CRS is geographic, project temporarily to UTM)
    if gdf.crs is not None and gdf.crs.is_geographic:
        metric_gdf = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = metric_gdf.geometry.area
    else:
        gdf["area_m2"] = gdf.geometry.area

    gdf["area_ha"] = gdf["area_m2"] / 10_000.0
    if min_area_ha > 0.0:
        gdf = gdf[gdf["area_ha"] >= min_area_ha].reset_index(drop=True)

    return gdf


def compute_statistics(
    lst_c: np.ndarray,
    ndvi: np.ndarray,
    hotspots_gdf: gpd.GeoDataFrame,
    scene_id: str = "unknown",
    acquisition_date: str = "unknown",
    valid_mask: np.ndarray | None = None,
) -> UHIStatistics:
    """Compute LST summary metrics and Pearson correlation between LST and NDVI.

    Args:
        lst_c: 2D float array of Land Surface Temperature (°C).
        ndvi: 2D float array of NDVI.
        hotspots_gdf: GeoDataFrame of extracted hotspot polygons.
        scene_id: Identifier of the analyzed Landsat scene.
        acquisition_date: Acquisition timestamp string of the scene.
        valid_mask: Optional boolean mask of valid pixels after QA_PIXEL masking.

    Returns:
        UHIStatistics dataclass containing all computed metrics.
    """
    is_valid_lst = np.isfinite(lst_c)
    if valid_mask is not None:
        is_valid_lst = is_valid_lst & valid_mask.astype(bool)

    valid_lst = lst_c[is_valid_lst]
    if valid_lst.size == 0:
        raise ValueError("No valid LST pixels available for statistical computation.")

    both_valid = is_valid_lst & np.isfinite(ndvi)
    lst_paired = lst_c[both_valid]
    ndvi_paired = ndvi[both_valid]

    if lst_paired.size >= 2 and np.std(lst_paired) > 0 and np.std(ndvi_paired) > 0:
        corr_val, p_val = pearsonr(lst_paired, ndvi_paired)
    else:
        corr_val, p_val = float("nan"), float("nan")

    hotspot_area_ha = (
        float(hotspots_gdf["area_ha"].sum()) if not hotspots_gdf.empty else 0.0
    )

    return UHIStatistics(
        scene_id=scene_id,
        acquisition_date=acquisition_date,
        min_lst_c=round(float(np.min(valid_lst)), 3),
        max_lst_c=round(float(np.max(valid_lst)), 3),
        mean_lst_c=round(float(np.mean(valid_lst)), 3),
        std_lst_c=round(float(np.std(valid_lst)), 3),
        median_lst_c=round(float(np.median(valid_lst)), 3),
        mean_ndvi=round(float(np.nanmean(ndvi[both_valid])), 4)
        if both_valid.any()
        else float("nan"),
        lst_ndvi_corr=round(float(corr_val), 4) if np.isfinite(corr_val) else float("nan"),
        lst_ndvi_p_value=float(p_val),
        valid_pixel_count=int(valid_lst.size),
        hotspot_area_ha=round(hotspot_area_ha, 2),
    )


def save_geotiff(
    array: np.ndarray,
    profile: Dict[str, Any],
    output_path: str | Path,
    dtype: str = "float32",
    nodata: float | int = np.nan,
) -> Path:
    """Write a 2D NumPy array to a single-band GeoTIFF file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": dtype,
            "nodata": nodata,
            "compress": "lzw",
        }
    )

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)

    logger.info("Saved GeoTIFF: %s", path)
    return path


def save_statistics_csv(stats: UHIStatistics, output_path: str | Path) -> Path:
    """Save UHIStatistics to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(stats)])
    df.to_csv(path, index=False)
    logger.info("Saved statistics CSV: %s", path)
    return path
