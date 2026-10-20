"""Unit tests for 5-class UHI classification, sieve hotspot vectorization, and statistics."""

from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from urban_heat_lst.analysis import (
    classify_lst,
    compute_statistics,
    extract_hotspot_vectors,
)
from urban_heat_lst.config import load_aoi, load_config


def test_classify_lst_five_classes_with_valid_mask() -> None:
    """Ensure μ and σ are computed strictly from post-masked valid pixels."""
    lst = np.array(
        [
            [18.0, 27.0, 30.0, 30.0, 30.0],
            [30.0, 30.0, 33.0, 42.0, 999.0],  # 999.0 is masked out by valid_mask
        ],
        dtype=np.float32,
    )
    valid_mask = np.array(
        [
            [True, True, True, True, True],
            [True, True, True, True, False],
        ]
    )
    classified, thresholds = classify_lst(lst, valid_mask=valid_mask)

    assert len(thresholds) == 4
    assert classified.dtype == np.uint8
    assert classified[0, 0] == 1  # 18.0 < μ - 1.5*σ
    assert classified[0, 2] == 3  # 30.0 is Moderate (μ)
    assert classified[1, 3] == 5  # 42.0 >= μ + 1.5*σ
    assert classified[1, 4] == 0  # Masked out -> 0 (NoData)


def test_extract_hotspot_vectors_with_sieve_and_statistics() -> None:
    """Verify rasterio.features.sieve removes isolated small hotspots (< min_area_ha)."""
    # Create a 5x5 grid (30m pixels -> each pixel is 0.09 ha; 6 pixels = 0.54 ha >= 0.5 ha)
    lst = np.full((5, 5), 30.0, dtype=np.float32)
    lst[0, :2] = 24.0
    # 6 contiguous hotspot pixels in bottom-right (6 * 0.09 ha = 0.54 ha)
    lst[3:5, 2:5] = 45.0
    # 1 isolated hotspot pixel in top-right (1 * 0.09 ha = 0.09 ha < 0.5 ha -> should be sieved out!)
    lst[0, 4] = 45.0

    ndvi = 0.6 - (lst - 24.0) * 0.02

    classified, _ = classify_lst(lst)
    profile = {
        "transform": from_origin(300000.0, 4100000.0, 30.0, 30.0),
        "crs": CRS.from_epsg(32636),
        "width": 5,
        "height": 5,
    }

    # With min_area_ha=0.5, the single isolated pixel (0.09 ha) is removed by sieve,
    # leaving only the 6-pixel contiguous cluster (6 * 900 m² = 5400 m² = 0.54 ha)
    hotspots = extract_hotspot_vectors(
        classified, profile, min_class=5, min_area_ha=0.5
    )
    assert len(hotspots) == 1
    assert np.isclose(hotspots["area_ha"].iloc[0], 0.54, atol=1e-2)

    stats = compute_statistics(
        lst_c=lst,
        ndvi=ndvi,
        hotspots_gdf=hotspots,
        scene_id="LC08_TEST",
        acquisition_date="2024-07-15",
    )
    assert stats.min_lst_c == 24.0
    assert stats.max_lst_c == 45.0
    assert stats.lst_ndvi_corr < -0.95
    assert stats.hotspot_area_ha == 0.54


def test_load_config_and_aoi() -> None:
    """Verify default config.yaml parameters (qa_mask_bits, season_months, min_area_ha) load properly."""
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config.yaml")
    assert cfg.max_cloud_cover == 15.0
    assert cfg.qa_mask_bits == [0, 1, 2, 3, 4, 5, 7]
    assert cfg.season_months == [6, 7, 8]
    assert cfg.min_area_ha == 0.5
    gdf, geom_dict = load_aoi(cfg.aoi_path)
    assert not gdf.empty
    assert geom_dict["type"] in ("Polygon", "MultiPolygon")
