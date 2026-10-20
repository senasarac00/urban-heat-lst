"""ESA WorldCover land cover download, 30m mode alignment, zonal LST stats, and hotspot attribution."""

import logging
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from shapely.geometry import mapping

logger = logging.getLogger(__name__)

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORLDCOVER_COLLECTION = "esa-worldcover"
BUILT_UP_CLASS = 50

# Official ESA WorldCover class codes, labels, and hex colors
WORLDCOVER_CLASSES: Dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

WORLDCOVER_COLORS: Dict[int, str] = {
    10: "#006400",  # Tree cover
    20: "#ffbb22",  # Shrubland
    30: "#ffff4c",  # Grassland
    40: "#f096ff",  # Cropland
    50: "#fa0000",  # Built-up
    60: "#b4b4b4",  # Bare / sparse vegetation
    70: "#f0f0f0",  # Snow and ice
    80: "#0064c8",  # Permanent water bodies
    90: "#0096a0",  # Herbaceous wetland
    95: "#00cf75",  # Mangroves
    100: "#fae6a0", # Moss and lichen
}


def _extract_item_year(item: pystac.Item) -> int:
    """Extract the representative year for an ESA WorldCover STAC Item."""
    if item.datetime is not None:
        return item.datetime.year
    start_dt = item.properties.get("start_datetime") or item.properties.get("datetime")
    if isinstance(start_dt, str) and len(start_dt) >= 4 and start_dt[:4].isdigit():
        return int(start_dt[:4])
    return 0


def search_latest_worldcover_items(aoi_geojson: Dict[str, Any]) -> list[pystac.Item]:
    """Query Planetary Computer for the latest year's ESA WorldCover tiles covering the AOI.

    Args:
        aoi_geojson: GeoJSON geometry dictionary (EPSG:4326) of the Area of Interest.

    Returns:
        List of signed STAC Items from the latest available ESA WorldCover year.

    Raises:
        RuntimeError: If no matching ESA WorldCover items are found.
    """
    logger.info("Searching ESA WorldCover (%s) on Planetary Computer...", WORLDCOVER_COLLECTION)
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[WORLDCOVER_COLLECTION],
        intersects=aoi_geojson,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No ESA WorldCover items found for the specified AOI.")

    latest_year = max(_extract_item_year(it) for it in items)
    latest_items = [it for it in items if _extract_item_year(it) == latest_year]

    logger.info(
        "Found %d ESA WorldCover items; selected %d tile(s) from latest year %d: %s",
        len(items),
        len(latest_items),
        latest_year,
        [it.id for it in latest_items],
    )
    return latest_items


def align_categorical_raster_mode(
    src_array: np.ndarray,
    src_transform: Any,
    src_crs: Any,
    dst_profile: Dict[str, Any],
) -> np.ndarray:
    """Reproject and align a 2D categorical array onto a target grid using mode resampling.

    Args:
        src_array: 2D uint8 categorical source array.
        src_transform: Affine transform of the source array.
        src_crs: CRS of the source array.
        dst_profile: Target rasterio profile with 'height', 'width', 'transform', and 'crs'.

    Returns:
        2D uint8 NumPy array aligned to dst_profile via Resampling.mode.
    """
    dst_shape = (int(dst_profile["height"]), int(dst_profile["width"]))
    dst_array = np.zeros(dst_shape, dtype=np.uint8)

    reproject(
        source=src_array.astype(np.uint8),
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.mode,
    )
    return dst_array


def fetch_and_align_worldcover(
    aoi_geojson: Dict[str, Any],
    target_profile: Dict[str, Any],
) -> np.ndarray:
    """Download ESA WorldCover for the AOI and align to the 30m LST grid using mode resampling.

    Args:
        aoi_geojson: GeoJSON geometry dictionary (EPSG:4326) of the AOI.
        target_profile: Rasterio profile of the 30m LST raster.

    Returns:
        2D uint8 NumPy array of ESA WorldCover classes aligned to target_profile.
    """
    items = search_latest_worldcover_items(aoi_geojson)
    dst_shape = (int(target_profile["height"]), int(target_profile["width"]))
    aligned = np.zeros(dst_shape, dtype=np.uint8)

    for item in items:
        asset_key = "map" if "map" in item.assets else list(item.assets.keys())[0]
        href = item.assets[asset_key].href
        logger.info("Warping ESA WorldCover tile %s to 30m LST grid (Resampling.mode)...", item.id)

        tile_dst = np.zeros(dst_shape, dtype=np.uint8)
        with rasterio.open(href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=tile_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.mode,
            )
        # Merge non-zero pixels into the output grid
        mask_nonzero = (aligned == 0) & (tile_dst > 0)
        aligned[mask_nonzero] = tile_dst[mask_nonzero]

    return aligned


def compute_landcover_lst_stats(
    lst_c: np.ndarray,
    worldcover: np.ndarray,
    valid_mask: np.ndarray | None = None,
    pixel_area_ha: float = 0.09,
) -> pd.DataFrame:
    """Compute mean, median, and standard deviation of LST for each ESA WorldCover class.

    Args:
        lst_c: 2D float array of Land Surface Temperature (°C).
        worldcover: 2D uint8 array of aligned ESA WorldCover classes.
        valid_mask: Optional boolean mask of valid clear pixels.
        pixel_area_ha: Area of a single pixel in hectares (default: 0.09 ha for 30m grid).

    Returns:
        DataFrame with columns: class_code, class_name, pixel_count, area_ha,
        mean_lst_c, median_lst_c, std_lst_c, min_lst_c, max_lst_c.
    """
    valid = np.isfinite(lst_c) & (worldcover > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.astype(bool)

    unique_codes = sorted(int(c) for c in np.unique(worldcover[valid]) if c > 0)
    rows = []
    for code in unique_codes:
        cls_mask = valid & (worldcover == code)
        vals = lst_c[cls_mask]
        if vals.size == 0:
            continue
        rows.append(
            {
                "class_code": code,
                "class_name": WORLDCOVER_CLASSES.get(code, f"Class {code}"),
                "pixel_count": int(vals.size),
                "area_ha": round(float(vals.size) * pixel_area_ha, 2),
                "mean_lst_c": round(float(np.mean(vals)), 3),
                "median_lst_c": round(float(np.median(vals)), 3),
                "std_lst_c": round(float(np.std(vals)), 3),
                "min_lst_c": round(float(np.min(vals)), 3),
                "max_lst_c": round(float(np.max(vals)), 3),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="mean_lst_c", ascending=False).reset_index(drop=True)
    return df


def enrich_hotspots_with_landcover(
    hotspots_gdf: gpd.GeoDataFrame,
    worldcover: np.ndarray,
    profile: Dict[str, Any],
) -> gpd.GeoDataFrame:
    """Add dominant land cover class and category percentages to each hotspot polygon.

    Added attributes:
      - dominant_lc_code: Integer code of the most frequent individual WorldCover class.
      - dominant_lc_name: Name of the dominant WorldCover class.
      - pct_urban: Percentage of polygon pixels in Built-up (50).
      - pct_agriculture: Percentage of polygon pixels in Cropland (40).
      - pct_bare: Percentage of polygon pixels in Bare / sparse vegetation (60).
      - pct_other: Percentage of polygon pixels in all other classes.

    Args:
        hotspots_gdf: GeoDataFrame of hotspot polygons.
        worldcover: 2D uint8 array of aligned ESA WorldCover classes.
        profile: Rasterio profile of the 30m grid.

    Returns:
        Updated GeoDataFrame with land cover attributes.
    """
    gdf = hotspots_gdf.copy()
    if gdf.empty:
        for col in [
            "dominant_lc_code",
            "dominant_lc_name",
            "pct_urban",
            "pct_agriculture",
            "pct_bare",
            "pct_other",
        ]:
            gdf[col] = []
        return gdf

    transform = profile["transform"]
    out_shape = worldcover.shape

    dom_codes = []
    dom_names = []
    pct_urban_list = []
    pct_agri_list = []
    pct_bare_list = []
    pct_other_list = []

    for geom in gdf.geometry:
        # Create boolean mask where True is inside the polygon
        inside_mask = ~geometry_mask(
            [mapping(geom)],
            out_shape=out_shape,
            transform=transform,
            invert=False,
            all_touched=False,
        )
        lc_vals = worldcover[inside_mask]
        lc_vals = lc_vals[lc_vals > 0]

        if lc_vals.size == 0:
            # Fallback with all_touched=True for very narrow polygons
            inside_mask = ~geometry_mask(
                [mapping(geom)],
                out_shape=out_shape,
                transform=transform,
                invert=False,
                all_touched=True,
            )
            lc_vals = worldcover[inside_mask]
            lc_vals = lc_vals[lc_vals > 0]

        if lc_vals.size == 0:
            dom_codes.append(0)
            dom_names.append("Unknown")
            pct_urban_list.append(0.0)
            pct_agri_list.append(0.0)
            pct_bare_list.append(0.0)
            pct_other_list.append(0.0)
            continue

        codes, counts = np.unique(lc_vals, return_counts=True)
        dom_idx = int(np.argmax(counts))
        dom_code = int(codes[dom_idx])
        total = float(lc_vals.size)

        urban_cnt = float(np.count_nonzero(lc_vals == 50))
        agri_cnt = float(np.count_nonzero(lc_vals == 40))
        bare_cnt = float(np.count_nonzero(lc_vals == 60))
        other_cnt = total - (urban_cnt + agri_cnt + bare_cnt)

        dom_codes.append(dom_code)
        dom_names.append(WORLDCOVER_CLASSES.get(dom_code, f"Class {dom_code}"))
        pct_urban_list.append(round((urban_cnt / total) * 100.0, 2))
        pct_agri_list.append(round((agri_cnt / total) * 100.0, 2))
        pct_bare_list.append(round((bare_cnt / total) * 100.0, 2))
        pct_other_list.append(round(max(0.0, (other_cnt / total) * 100.0), 2))

    gdf["dominant_lc_code"] = dom_codes
    gdf["dominant_lc_name"] = dom_names
    gdf["pct_urban"] = pct_urban_list
    gdf["pct_agriculture"] = pct_agri_list
    gdf["pct_bare"] = pct_bare_list
    gdf["pct_other"] = pct_other_list

    return gdf


def compute_hotspot_landcover_distribution(
    hotspots_gdf: gpd.GeoDataFrame,
    worldcover: np.ndarray,
    profile: Dict[str, Any],
) -> pd.DataFrame:
    """Compute the distribution of total hotspot area across ESA WorldCover classes.

    Args:
        hotspots_gdf: GeoDataFrame of extracted hotspot polygons.
        worldcover: 2D uint8 array of aligned ESA WorldCover classes.
        profile: Rasterio profile of the 30m grid.

    Returns:
        DataFrame with columns: class_code, class_name, pixel_count, area_ha, pct_of_hotspots.
    """
    transform = profile["transform"]
    pixel_area_ha = abs(float(transform.a) * float(transform.e)) / 10_000.0
    if pixel_area_ha <= 0:
        pixel_area_ha = 0.09

    if hotspots_gdf.empty:
        return pd.DataFrame(
            columns=["class_code", "class_name", "pixel_count", "area_ha", "pct_of_hotspots"]
        )

    inside_hotspots = ~geometry_mask(
        [mapping(geom) for geom in hotspots_gdf.geometry],
        out_shape=worldcover.shape,
        transform=transform,
        invert=False,
    )
    lc_hotspot_pixels = worldcover[inside_hotspots]
    lc_hotspot_pixels = lc_hotspot_pixels[lc_hotspot_pixels > 0]

    total_pixels = max(int(lc_hotspot_pixels.size), 1)
    codes, counts = np.unique(lc_hotspot_pixels, return_counts=True)

    rows = []
    for code, count in zip(codes, counts):
        code_int = int(code)
        cnt_int = int(count)
        rows.append(
            {
                "class_code": code_int,
                "class_name": WORLDCOVER_CLASSES.get(code_int, f"Class {code_int}"),
                "pixel_count": cnt_int,
                "area_ha": round(cnt_int * pixel_area_ha, 2),
                "pct_of_hotspots": round((cnt_int / total_pixels) * 100.0, 2),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="area_ha", ascending=False).reset_index(drop=True)
    return df
