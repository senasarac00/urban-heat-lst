"""Unit tests for ESA WorldCover alignment, zonal LST statistics, urban_only mode, and hotspot attribution."""

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from urban_heat_lst.analysis import classify_lst, extract_hotspot_vectors
from urban_heat_lst.landcover import (
    align_categorical_raster_mode,
    compute_hotspot_landcover_distribution,
    compute_landcover_lst_stats,
    enrich_hotspots_with_landcover,
)


def test_align_categorical_raster_mode() -> None:
    """Verify mode resampling preserves categorical class values when downsampling 10m to 30m."""
    # Create a 6x6 10m raster -> 2x2 30m target grid
    src_10m = np.full((6, 6), 10, dtype=np.uint8)
    # In top-left 3x3 block (9 pixels), set 6 pixels to 50 (Built-up) and 3 pixels to 40 (Cropland)
    # Majority/mode in top-left 30m pixel must be 50!
    src_10m[0:2, 0:3] = 50
    src_10m[2, 0:3] = 40
    # In bottom-right 3x3 block, set 7 pixels to 40 (Cropland)
    src_10m[3:6, 3:6] = 40

    src_transform = from_origin(300000.0, 4100000.0, 10.0, 10.0)
    dst_profile = {
        "height": 2,
        "width": 2,
        "transform": from_origin(300000.0, 4100000.0, 30.0, 30.0),
        "crs": CRS.from_epsg(32636),
    }

    aligned = align_categorical_raster_mode(
        src_10m, src_transform, CRS.from_epsg(32636), dst_profile
    )
    assert aligned.shape == (2, 2)
    assert aligned[0, 0] == 50
    assert aligned[1, 1] == 40


def test_compute_landcover_lst_stats() -> None:
    """Verify per-class mean, median, and std LST calculations."""
    lst = np.array([[45.0, 47.0], [32.0, 34.0]], dtype=np.float32)
    wc = np.array([[50, 50], [10, 10]], dtype=np.uint8)

    df = compute_landcover_lst_stats(lst, wc)
    assert len(df) == 2
    # Sorted descending by mean_lst_c -> Built-up (50) first
    assert df.iloc[0]["class_code"] == 50
    assert df.iloc[0]["class_name"] == "Built-up"
    assert np.isclose(df.iloc[0]["mean_lst_c"], 46.0)
    assert np.isclose(df.iloc[0]["median_lst_c"], 46.0)
    assert np.isclose(df.iloc[0]["std_lst_c"], 1.0)


def test_urban_only_classification_and_hotspot_enrichment() -> None:
    """Verify urban_only=True restricts μ/σ and classification to Built-up (50) and enriches hotspots."""
    # 4x4 grid where top 2 rows are Built-up (50) and bottom 2 rows are Bare (60) with extreme 60°C
    lst = np.array(
        [
            [35.0, 38.0, 40.0, 40.0],
            [40.0, 40.0, 42.0, 48.0],
            [60.0, 60.0, 60.0, 60.0],
            [60.0, 60.0, 60.0, 60.0],
        ],
        dtype=np.float32,
    )
    wc = np.array(
        [
            [50, 50, 50, 50],
            [50, 50, 50, 50],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
        ],
        dtype=np.uint8,
    )

    classified_urban, _ = classify_lst(lst, worldcover=wc, urban_only=True)
    # Non-urban rows (rows 2 and 3) must be 0 (NoData) when urban_only=True
    assert np.all(classified_urban[2:, :] == 0)
    # Hottest urban pixel (48.0 at [1, 3]) must be Class 5
    assert classified_urban[1, 3] == 5

    profile = {
        "transform": from_origin(300000.0, 4100000.0, 30.0, 30.0),
        "crs": CRS.from_epsg(32636),
        "width": 4,
        "height": 4,
    }
    hotspots = extract_hotspot_vectors(
        classified_urban, profile, min_class=5, min_area_ha=0.05
    )
    enriched = enrich_hotspots_with_landcover(hotspots, wc, profile)
    assert len(enriched) == 1
    assert enriched.iloc[0]["dominant_lc_code"] == 50
    assert enriched.iloc[0]["dominant_lc_name"] == "Built-up"
    assert enriched.iloc[0]["pct_urban"] == 100.0

    dist_df = compute_hotspot_landcover_distribution(enriched, wc, profile)
    assert len(dist_df) == 1
    assert dist_df.iloc[0]["class_code"] == 50
    assert dist_df.iloc[0]["pct_of_hotspots"] == 100.0
