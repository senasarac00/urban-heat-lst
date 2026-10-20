#!/usr/bin/env python3
"""Single-command CLI runner for the urban-heat-lst pipeline."""

import argparse
import logging
from pathlib import Path
import sys

from urban_heat_lst.analysis import (
    classify_lst,
    compute_statistics,
    extract_hotspot_vectors,
    save_geotiff,
    save_statistics_csv,
)
from urban_heat_lst.config import load_aoi, load_config
from urban_heat_lst.indices import compute_lst_celsius, compute_ndvi
from urban_heat_lst.preprocessing import load_and_preprocess_scene
from urban_heat_lst.stac_search import search_best_landsat_scene
from urban_heat_lst.visualization import generate_all_maps


def setup_logging() -> None:
    """Configure console logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_pipeline(config_path: str | Path = "config.yaml") -> dict[str, Path]:
    """Execute the complete end-to-end Urban Heat Island analysis workflow.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary mapping output artifact keys to their saved file Paths.
    """
    logger = logging.getLogger("urban_heat_lst.runner")

    # 1. Load configuration and AOI
    logger.info("Step 1/6: Loading configuration from %s", config_path)
    cfg = load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    aoi_gdf, aoi_geojson = load_aoi(cfg.aoi_path)
    logger.info("Loaded AOI from %s", cfg.aoi_path)

    # 2. Search Planetary Computer STAC for the least-cloudy Landsat 8/9 scene with season filter
    logger.info(
        "Step 2/6: Searching Landsat 8/9 Collection 2 Level-2 scenes (%s, months=%s, cloud < %.1f%%)",
        cfg.date_range,
        cfg.season_months,
        cfg.max_cloud_cover,
    )
    best_item = search_best_landsat_scene(
        aoi_geojson=aoi_geojson,
        date_range=cfg.date_range,
        max_cloud_cover=cfg.max_cloud_cover,
        season_months=cfg.season_months,
    )

    # 3. Apply QA_PIXEL bitmask (clouds, shadows, snow, water) and clip to AOI
    logger.info(
        "Step 3/6: Applying QA_PIXEL mask (bits=%s) and clipping to AOI",
        cfg.qa_mask_bits,
    )
    clipped = load_and_preprocess_scene(
        item=best_item,
        aoi_gdf=aoi_gdf,
        qa_mask_bits=cfg.qa_mask_bits,
    )

    # 4. Compute LST in Celsius and NDVI
    logger.info("Step 4/6: Computing LST (°C) from ST_B10 and NDVI from SR_B4/SR_B5")
    lst_c = compute_lst_celsius(clipped.st_b10)
    ndvi = compute_ndvi(clipped.sr_b4, clipped.sr_b5)

    # 5. Classify LST (μ ± σ from post-masked valid pixels), sieve, and extract hotspot polygons
    logger.info(
        "Step 5/6: Classifying LST (μ ± σ of valid pixels) and extracting hotspots (min_area_ha=%.2f)",
        cfg.min_area_ha,
    )
    classified_lst, thresholds = classify_lst(lst_c, valid_mask=clipped.valid_mask)
    logger.info(
        "LST Class Thresholds (°C): [%.2f, %.2f, %.2f, %.2f]",
        thresholds[0],
        thresholds[1],
        thresholds[2],
        thresholds[3],
    )

    hotspots_gdf = extract_hotspot_vectors(
        classified_lst=classified_lst,
        profile=clipped.profile,
        min_class=cfg.hotspot_min_class,
        min_area_ha=cfg.min_area_ha,
    )

    acq_date = str(
        best_item.properties.get(
            "datetime",
            best_item.datetime.isoformat() if best_item.datetime else "unknown",
        )
    )
    stats = compute_statistics(
        lst_c=lst_c,
        ndvi=ndvi,
        hotspots_gdf=hotspots_gdf,
        scene_id=best_item.id,
        acquisition_date=acq_date,
        valid_mask=clipped.valid_mask,
    )

    # 6. Export GeoTIFFs, GeoPackage, CSV, and PNG maps
    logger.info("Step 6/6: Exporting rasters, hotspot vectors, statistics CSV, and maps")
    lst_tif_path = save_geotiff(
        lst_c,
        clipped.profile,
        cfg.output_dir / "lst_celsius.tif",
        dtype="float32",
        nodata=float("nan"),
    )
    class_tif_path = save_geotiff(
        classified_lst,
        clipped.profile,
        cfg.output_dir / "lst_classified.tif",
        dtype="uint8",
        nodata=0,
    )

    hotspots_path = cfg.output_dir / "hotspots.gpkg"
    hotspots_gdf.to_file(hotspots_path, driver="GPKG", layer="hotspots")
    logger.info(
        "Saved %d hotspot polygons (%.2f ha) to %s",
        len(hotspots_gdf),
        stats.hotspot_area_ha,
        hotspots_path,
    )

    csv_path = save_statistics_csv(stats, cfg.output_dir / "statistics.csv")
    map_paths = generate_all_maps(
        lst_c=lst_c,
        ndvi=ndvi,
        classified_lst=classified_lst,
        profile=clipped.profile,
        stats=stats,
        output_dir=cfg.output_dir,
    )

    logger.info(
        "Pipeline completed! LST Min=%.2f°C, Max=%.2f°C, Mean=%.2f°C, LST-NDVI r=%.4f",
        stats.min_lst_c,
        stats.max_lst_c,
        stats.mean_lst_c,
        stats.lst_ndvi_corr,
    )

    return {
        "lst_geotiff": lst_tif_path,
        "classified_geotiff": class_tif_path,
        "hotspots_gpkg": hotspots_path,
        "statistics_csv": csv_path,
        **map_paths,
    }


def main() -> int:
    """Parse CLI arguments and invoke the pipeline."""
    parser = argparse.ArgumentParser(
        description="urban-heat-lst: Urban Heat Island analysis tool using Landsat 8/9 C2 L2."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    setup_logging()
    try:
        run_pipeline(args.config)
    except Exception as exc:
        logging.getLogger("urban_heat_lst").exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
