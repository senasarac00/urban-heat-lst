"""Configuration and AOI loader module for urban-heat-lst."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import yaml
from shapely.geometry import mapping

DEFAULT_QA_MASK_BITS = [0, 1, 2, 3, 4, 5, 7]
DEFAULT_SEASON_MONTHS = [6, 7, 8]


@dataclass
class PipelineConfig:
    """Data structure holding validated pipeline configuration parameters."""

    aoi_path: Path
    date_range: str
    max_cloud_cover: float
    qa_mask_bits: list[int] = field(default_factory=lambda: list(DEFAULT_QA_MASK_BITS))
    season_months: list[int] = field(default_factory=lambda: list(DEFAULT_SEASON_MONTHS))
    hotspot_min_class: int = 5
    min_area_ha: float = 0.5
    output_dir: Path = Path("outputs")


def load_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """Load and validate pipeline settings from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        PipelineConfig instance with validated parameters.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    required_keys = ["aoi_path", "date_range", "max_cloud_cover"]
    for key in required_keys:
        if key not in raw:
            raise ValueError(f"Missing required configuration key: '{key}'")

    aoi_path = Path(raw["aoi_path"])
    if not aoi_path.is_absolute():
        aoi_path = (path.parent / aoi_path).resolve()

    output_dir = Path(raw.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()

    max_cloud = float(raw["max_cloud_cover"])
    if not (0.0 <= max_cloud <= 100.0):
        raise ValueError("max_cloud_cover must be between 0 and 100.")

    raw_bits = raw.get("qa_mask_bits", DEFAULT_QA_MASK_BITS)
    if not isinstance(raw_bits, list) or not all(
        isinstance(b, int) and 0 <= b <= 15 for b in raw_bits
    ):
        raise ValueError("qa_mask_bits must be a list of integers between 0 and 15.")
    qa_mask_bits = [int(b) for b in raw_bits]

    raw_months = raw.get("season_months", DEFAULT_SEASON_MONTHS)
    if not isinstance(raw_months, list) or not all(
        isinstance(m, int) and 1 <= m <= 12 for m in raw_months
    ):
        raise ValueError("season_months must be a list of month integers between 1 and 12.")
    season_months = [int(m) for m in raw_months]

    hotspot_min_class = int(raw.get("hotspot_min_class", 5))
    if not (1 <= hotspot_min_class <= 5):
        raise ValueError("hotspot_min_class must be between 1 and 5.")

    min_area_ha = float(raw.get("min_area_ha", 0.5))
    if min_area_ha < 0.0:
        raise ValueError("min_area_ha must be non-negative.")

    return PipelineConfig(
        aoi_path=aoi_path,
        date_range=str(raw["date_range"]),
        max_cloud_cover=max_cloud,
        qa_mask_bits=qa_mask_bits,
        season_months=season_months,
        hotspot_min_class=hotspot_min_class,
        min_area_ha=min_area_ha,
        output_dir=output_dir,
    )


def load_aoi(aoi_path: str | Path) -> tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Load the Area of Interest (AOI) GeoJSON and convert to EPSG:4326.

    Args:
        aoi_path: Path to the AOI vector file (GeoJSON).

    Returns:
        Tuple of (GeoDataFrame in EPSG:4326, GeoJSON geometry dict of the unified AOI).

    Raises:
        FileNotFoundError: If the AOI file does not exist.
        ValueError: If the AOI file contains no valid geometries.
    """
    path = Path(aoi_path)
    if not path.exists():
        raise FileNotFoundError(f"AOI file not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI file contains no features: {path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    unified_geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    geom_dict = mapping(unified_geom)

    return gdf, geom_dict
