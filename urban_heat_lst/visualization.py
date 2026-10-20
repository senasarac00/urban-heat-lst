"""Publication-quality PNG map and correlation plot generation using Matplotlib."""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.landcover import WORLDCOVER_CLASSES, WORLDCOVER_COLORS
from urban_heat_lst.analysis import CLASS_LABELS, UHIStatistics

logger = logging.getLogger(__name__)

# Color palette for the 5 UHI classes (Cool Blue -> Cyan -> Yellow -> Orange -> Deep Red)
UHI_CLASS_COLORS = [
    "#2c7bb6",  # Class 1: Very Low
    "#abd9e9",  # Class 2: Low
    "#ffffbf",  # Class 3: Moderate
    "#fdae61",  # Class 4: High
    "#d7191c",  # Class 5: Very High (Hotspot)
]


def _get_raster_extent(profile: Dict[str, Any]) -> list[float]:
    """Compute [left, right, bottom, top] spatial extent from a rasterio profile."""
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e
    return [left, right, bottom, top]


def plot_lst_map(
    lst_c: np.ndarray,
    profile: Dict[str, Any],
    stats: UHIStatistics,
    output_path: str | Path,
) -> Path:
    """Render and save a continuous Land Surface Temperature (°C) map."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = _get_raster_extent(profile)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="#e6e6e6")

    vmin = float(np.nanpercentile(lst_c, 2))
    vmax = float(np.nanpercentile(lst_c, 98))

    im = ax.imshow(lst_c, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Land Surface Temperature (°C)", fontsize=11)

    ax.set_title(
        f"Land Surface Temperature (LST) — {stats.scene_id}\n"
        f"Date: {stats.acquisition_date[:10]} | Mean: {stats.mean_lst_c:.1f}°C | "
        f"Min: {stats.min_lst_c:.1f}°C | Max: {stats.max_lst_c:.1f}°C",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved LST map: %s", path)
    return path


def plot_classified_uhi_map(
    classified_lst: np.ndarray,
    profile: Dict[str, Any],
    stats: UHIStatistics,
    output_path: str | Path,
) -> Path:
    """Render and save the 5-class Urban Heat Island map with class area percentages."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = _get_raster_extent(profile)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    masked_class = np.ma.masked_where(classified_lst == 0, classified_lst)
    cmap = mcolors.ListedColormap(UHI_CLASS_COLORS)
    cmap.set_bad(color="#e6e6e6")
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(masked_class, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")

    total_valid = max(int(np.count_nonzero(classified_lst > 0)), 1)
    legend_patches = []
    for class_id in range(1, 6):
        count = int(np.count_nonzero(classified_lst == class_id))
        pct = count / total_valid * 100.0
        label = f"Class {class_id}: {CLASS_LABELS[class_id]} ({pct:.1f}%)"
        legend_patches.append(
            mpatches.Patch(color=UHI_CLASS_COLORS[class_id - 1], label=label)
        )

    ax.legend(
        handles=legend_patches,
        loc="lower left",
        framealpha=0.92,
        fontsize=8.5,
        title="UHI Intensity Classes (Mean ± SD)",
        title_fontsize=9.5,
    )

    ax.set_title(
        f"Urban Heat Island (UHI) 5-Class Classification\n"
        f"Scene: {stats.scene_id} | Extreme Hotspot Area (Class 5): {stats.hotspot_area_ha:.1f} ha",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved UHI classified map: %s", path)
    return path


def plot_ndvi_map(
    ndvi: np.ndarray,
    profile: Dict[str, Any],
    stats: UHIStatistics,
    output_path: str | Path,
) -> Path:
    """Render and save a Normalized Difference Vegetation Index (NDVI) map."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = _get_raster_extent(profile)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad(color="#e6e6e6")

    im = ax.imshow(ndvi, cmap=cmap, vmin=-0.1, vmax=0.8, extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI", fontsize=11)

    ax.set_title(
        f"Normalized Difference Vegetation Index (NDVI)\n"
        f"Scene: {stats.scene_id} | Mean NDVI: {stats.mean_ndvi:.3f}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved NDVI map: %s", path)
    return path


def plot_lst_ndvi_correlation(
    lst_c: np.ndarray,
    ndvi: np.ndarray,
    stats: UHIStatistics,
    output_path: str | Path,
) -> Path:
    """Render and save a 2D hexbin density plot of LST vs. NDVI with linear regression."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    valid = np.isfinite(lst_c) & np.isfinite(ndvi)
    x = ndvi[valid]
    y = lst_c[valid]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    hb = ax.hexbin(x, y, gridsize=60, cmap="inferno", mincnt=1, bins="log")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(Pixel Count)", fontsize=10)

    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color="#00ffff",
            linewidth=2.2,
            linestyle="--",
            label=f"Linear fit: LST = {slope:.2f}·NDVI + {intercept:.2f}\nPearson r = {stats.lst_ndvi_corr:.3f}",
        )
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    ax.set_title(
        f"LST vs. NDVI Correlation — {stats.scene_id}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("NDVI", fontsize=11)
    ax.set_ylabel("Land Surface Temperature (°C)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved LST-NDVI correlation plot: %s", path)
    return path


def plot_lst_landcover_comparison(
    lst_c: np.ndarray,
    worldcover: np.ndarray,
    profile: Dict[str, Any],
    stats: UHIStatistics,
    output_path: str | Path,
) -> Path:
    """Render a side-by-side comparison map of Land Surface Temperature (°C) and ESA WorldCover."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = _get_raster_extent(profile)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=200)

    # Left panel: Continuous LST
    cmap_lst = plt.get_cmap("RdYlBu_r").copy()
    cmap_lst.set_bad(color="#e6e6e6")
    vmin = float(np.nanpercentile(lst_c, 2))
    vmax = float(np.nanpercentile(lst_c, 98))
    im1 = ax1.imshow(lst_c, cmap=cmap_lst, vmin=vmin, vmax=vmax, extent=extent)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Land Surface Temperature (°C)", fontsize=10)
    ax1.set_title(
        f"Land Surface Temperature (LST °C)\nMean: {stats.mean_lst_c:.1f}°C | Max: {stats.max_lst_c:.1f}°C",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Easting (m)", fontsize=10)
    ax1.set_ylabel("Northing (m)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Right panel: Categorical ESA WorldCover
    codes = sorted(WORLDCOVER_CLASSES.keys())
    idx_map = np.full(worldcover.shape, -1, dtype=np.int16)
    for idx, code in enumerate(codes):
        idx_map[worldcover == code] = idx

    masked_lc = np.ma.masked_where(idx_map < 0, idx_map)
    lc_colors = [WORLDCOVER_COLORS[c] for c in codes]
    cmap_lc = mcolors.ListedColormap(lc_colors)
    cmap_lc.set_bad(color="#e6e6e6")
    bounds = [i - 0.5 for i in range(len(codes) + 1)]
    norm_lc = mcolors.BoundaryNorm(bounds, cmap_lc.N)

    ax2.imshow(masked_lc, cmap=cmap_lc, norm=norm_lc, extent=extent, interpolation="nearest")

    present_codes = sorted(int(c) for c in np.unique(worldcover) if c in WORLDCOVER_CLASSES)
    total_lc = max(int(np.count_nonzero(worldcover > 0)), 1)
    patches = []
    for code in present_codes:
        cnt = int(np.count_nonzero(worldcover == code))
        pct = (cnt / total_lc) * 100.0
        patches.append(
            mpatches.Patch(
                color=WORLDCOVER_COLORS[code],
                label=f"{code} - {WORLDCOVER_CLASSES[code]} ({pct:.1f}%)",
            )
        )

    ax2.legend(
        handles=patches,
        loc="lower left",
        framealpha=0.92,
        fontsize=8.5,
        title="ESA WorldCover (30m Mode Aligned)",
        title_fontsize=9.5,
    )
    ax2.set_title(
        "ESA WorldCover Land Cover Classification\n(Resampled to 30m Grid via Mode)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Easting (m)", fontsize=10)
    ax2.set_ylabel("Northing (m)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Land Surface Temperature vs. ESA WorldCover — {stats.scene_id} ({stats.acquisition_date[:10]})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved LST vs Land Cover comparison map: %s", path)
    return path


def plot_hotspot_landcover_bar(
    dist_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Render and save a bar chart showing total hotspot area distribution across land cover classes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    if dist_df.empty:
        ax.text(0.5, 0.5, "No Hotspots Found", ha="center", va="center", fontsize=12)
    else:
        labels = [
            f"{row['class_name']}\n(Code {int(row['class_code'])})"
            for _, row in dist_df.iterrows()
        ]
        areas = dist_df["area_ha"].to_numpy(dtype=float)
        pcts = dist_df["pct_of_hotspots"].to_numpy(dtype=float)
        colors = [
            WORLDCOVER_COLORS.get(int(c), "#888888") for c in dist_df["class_code"]
        ]

        bars = ax.bar(labels, areas, color=colors, edgecolor="#333333", width=0.58)
        max_h = max(float(np.max(areas)), 1.0)
        ax.set_ylim(0, max_h * 1.18)

        for bar, area_val, pct_val in zip(bars, areas, pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_h * 0.02,
                f"{area_val:.1f} ha\n({pct_val:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_title(
        "Distribution of Thermal Hotspot Area Across ESA WorldCover Classes",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Hotspot Area (ha)", fontsize=11)
    ax.set_xlabel("ESA WorldCover Class", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hotspot land cover distribution bar chart: %s", path)
    return path


def generate_all_maps(
    lst_c: np.ndarray,
    ndvi: np.ndarray,
    classified_lst: np.ndarray,
    profile: Dict[str, Any],
    stats: UHIStatistics,
    output_dir: str | Path,
    worldcover: np.ndarray | None = None,
    hotspot_lc_dist_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Generate all PNG visualizations and return their file paths."""
    maps_dir = Path(output_dir) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "lst_map": plot_lst_map(lst_c, profile, stats, maps_dir / "lst_map.png"),
        "uhi_classified_map": plot_classified_uhi_map(
            classified_lst, profile, stats, maps_dir / "uhi_classified_map.png"
        ),
        "ndvi_map": plot_ndvi_map(ndvi, profile, stats, maps_dir / "ndvi_map.png"),
        "lst_ndvi_correlation": plot_lst_ndvi_correlation(
            lst_c, ndvi, stats, maps_dir / "lst_ndvi_correlation.png"
        ),
    }

    if worldcover is not None:
        result["lst_landcover_comparison"] = plot_lst_landcover_comparison(
            lst_c,
            worldcover,
            profile,
            stats,
            maps_dir / "lst_landcover_comparison.png",
        )

    if hotspot_lc_dist_df is not None:
        result["hotspot_landcover_bar"] = plot_hotspot_landcover_bar(
            hotspot_lc_dist_df,
            maps_dir / "hotspot_landcover_distribution.png",
        )

    return result
