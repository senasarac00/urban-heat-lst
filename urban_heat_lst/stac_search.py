"""Microsoft Planetary Computer STAC search and Landsat 8/9 scene selection."""

from datetime import datetime
import logging
from typing import Any, Dict, Sequence

import planetary_computer
import pystac
import pystac_client
from shapely.geometry import shape

logger = logging.getLogger(__name__)

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "landsat-c2-l2"
SUPPORTED_PLATFORMS = ["landsat-8", "landsat-9"]
DEFAULT_SEASON_MONTHS = (6, 7, 8)


def compute_aoi_coverage(item: pystac.Item, aoi_geojson: Dict[str, Any]) -> float:
    """Calculate the fraction of the AOI covered by the STAC item footprint.

    Args:
        item: STAC Item with a valid geometry footprint.
        aoi_geojson: GeoJSON geometry dictionary in EPSG:4326.

    Returns:
        Float between 0.0 and 1.0 representing the intersection coverage ratio.
    """
    if item.geometry is None:
        return 0.0
    aoi_poly = shape(aoi_geojson)
    if aoi_poly.is_empty or aoi_poly.area == 0:
        return 0.0
    scene_poly = shape(item.geometry)
    intersection_area = aoi_poly.intersection(scene_poly).area
    return float(intersection_area / aoi_poly.area)


def _extract_item_month(item: pystac.Item) -> int | None:
    """Extract the acquisition month (1-12) from a STAC Item."""
    if item.datetime is not None:
        return item.datetime.month
    dt_str = item.properties.get("datetime")
    if isinstance(dt_str, str) and dt_str:
        parsed = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return parsed.month
    return None


def filter_items_by_season(
    items: Sequence[pystac.Item],
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
) -> list[pystac.Item]:
    """Filter STAC Items to only those acquired within the specified months.

    Args:
        items: Sequence of candidate STAC Items.
        season_months: Allowed calendar months (1=Jan .. 12=Dec). Defaults to (6, 7, 8) for June-August.

    Returns:
        List of STAC Items matching the seasonal filter.
    """
    if not season_months:
        return list(items)
    allowed = set(int(m) for m in season_months)
    return [item for item in items if _extract_item_month(item) in allowed]


def select_least_cloudy_item(
    items: Sequence[pystac.Item],
    aoi_geojson: Dict[str, Any] | None = None,
    season_months: Sequence[int] | None = DEFAULT_SEASON_MONTHS,
    min_coverage: float = 0.85,
) -> pystac.Item:
    """Select the least cloudy Landsat scene matching the season filter and AOI coverage.

    Args:
        items: Sequence of candidate STAC Items.
        aoi_geojson: Optional GeoJSON geometry dictionary to verify spatial coverage.
        season_months: Allowed calendar months (default: June-August [6, 7, 8]).
        min_coverage: Minimum fraction of the AOI that must be covered by the scene.

    Returns:
        The best matching STAC Item with the lowest cloud cover.

    Raises:
        ValueError: If no items match the criteria.
    """
    if not items:
        raise ValueError("No STAC items provided for scene selection.")

    candidates = list(items)
    if season_months:
        candidates = filter_items_by_season(candidates, season_months)
        if not candidates:
            raise ValueError(
                f"No STAC items match the specified season_months={list(season_months)}."
            )

    if aoi_geojson is not None:
        well_covered = [
            item
            for item in candidates
            if compute_aoi_coverage(item, aoi_geojson) >= min_coverage
        ]
        if well_covered:
            candidates = well_covered
        else:
            logger.warning(
                "No single scene covers >= %.0f%% of the AOI; falling back to highest coverage scenes.",
                min_coverage * 100,
            )

    def sort_key(item: pystac.Item) -> tuple[float, float]:
        cloud_cover = float(item.properties.get("eo:cloud_cover", 100.0))
        coverage = (
            compute_aoi_coverage(item, aoi_geojson) if aoi_geojson is not None else 1.0
        )
        return (cloud_cover, -coverage)

    sorted_items = sorted(candidates, key=sort_key)
    return sorted_items[0]


def search_best_landsat_scene(
    aoi_geojson: Dict[str, Any],
    date_range: str,
    max_cloud_cover: float,
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
) -> pystac.Item:
    """Query Microsoft Planetary Computer for Landsat 8/9 C2 L2 and pick the best scene.

    Args:
        aoi_geojson: GeoJSON geometry dictionary (EPSG:4326) of the Area of Interest.
        date_range: ISO 8601 date range string (e.g., '2024-01-01/2024-12-31').
        max_cloud_cover: Maximum allowed scene cloud cover percentage (0-100).
        season_months: Calendar months to include (default: [6, 7, 8] for June-August).

    Returns:
        Signed STAC Item corresponding to the least-cloudy matching scene.

    Raises:
        RuntimeError: If no matching scenes are found.
    """
    logger.info(
        "Connecting to Microsoft Planetary Computer STAC API (%s)...", STAC_API_URL
    )
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[COLLECTION_ID],
        intersects=aoi_geojson,
        datetime=date_range,
        query={
            "eo:cloud_cover": {"lt": max_cloud_cover},
            "platform": {"in": SUPPORTED_PLATFORMS},
        },
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Landsat 8/9 Collection 2 Level-2 scenes found for date_range='{date_range}' "
            f"with eo:cloud_cover < {max_cloud_cover}%."
        )

    seasonal_items = filter_items_by_season(items, season_months)
    if not seasonal_items:
        raise RuntimeError(
            f"Found {len(items)} scenes in date_range='{date_range}', but none fell within "
            f"season_months={list(season_months)}."
        )

    best_item = select_least_cloudy_item(
        seasonal_items,
        aoi_geojson=aoi_geojson,
        season_months=season_months,
    )
    cloud_val = best_item.properties.get("eo:cloud_cover")
    platform = best_item.properties.get("platform")
    acq_date = best_item.properties.get("datetime", best_item.datetime)
    coverage = compute_aoi_coverage(best_item, aoi_geojson) * 100.0

    logger.info(
        "Found %d total scenes (%d in season_months=%s). Selected scene: %s "
        "(Platform: %s, Date: %s, Cloud Cover: %.2f%%, AOI Coverage: %.1f%%)",
        len(items),
        len(seasonal_items),
        list(season_months),
        best_item.id,
        platform,
        acq_date,
        float(cloud_val) if cloud_val is not None else -1.0,
        coverage,
    )

    return best_item
