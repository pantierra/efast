# -*- coding: utf-8 -*-
"""
Cloud-native Sentinel-2 download for EFAST.

Provides STAC-based windowed COG reads as a self-contained alternative to
the full-SAFE download in run_efast.py without modifying any existing module.

S2 L2A products are accessed via the AWS Element84 Earth Search catalogue
(https://earth-search.aws.element84.com/v1), which hosts Cloud-Optimised
GeoTIFFs. Only the bytes covering the AOI window are transferred.

Key functions
-------------
wkt_to_bbox        : convert WKT geometry to [W, S, E, N] bounding box
stac_search_s2     : search Earth Search for S2 L2A items intersecting a bbox
download_s2_window : range-read, mask and write windowed REFL GeoTIFFs
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from pystac_client import Client
from rasterio.warp import transform_geom
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from scipy.ndimage import zoom
from shapely import wkt as shapely_wkt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

# Earth Search asset keys for each S2 band name
_BAND_ASSETS: dict[str, str] = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
}
_SCL_ASSET = "scl"

# Minimum half-width (degrees) used to expand POINT geometries to a valid bbox
_MIN_BBOX_HALF_DEG = 0.008

# EFAST block aggregation expects HR dimensions divisible by this ratio
_EFAST_RATIO_ALIGN = 30


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def wkt_to_bbox(geometry_wkt: str) -> list[float]:
    """Convert a WKT geometry to a ``[west, south, east, north]`` bbox.

    Parameters
    ----------
    geometry_wkt : str
        WKT geometry, e.g. ``POINT (-15.4 15.4)`` or a polygon string.

    Returns
    -------
    list[float]
        ``[west, south, east, north]`` in EPSG:4326. POINT geometries are
        expanded by ``_MIN_BBOX_HALF_DEG`` on every side so the bbox is
        non-degenerate.
    """
    geom = shapely_wkt.loads(geometry_wkt)
    minx, miny, maxx, maxy = geom.bounds
    if minx == maxx and miny == maxy:
        minx -= _MIN_BBOX_HALF_DEG
        maxx += _MIN_BBOX_HALF_DEG
        miny -= _MIN_BBOX_HALF_DEG
        maxy += _MIN_BBOX_HALF_DEG
    return [minx, miny, maxx, maxy]


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------


def stac_search_s2(
    bbox: list[float],
    start_date: datetime,
    end_date: datetime,
) -> list[Any]:
    """Search Earth Search for Sentinel-2 L2A items intersecting a bbox.

    Parameters
    ----------
    bbox : list[float]
        ``[west, south, east, north]`` in EPSG:4326.
    start_date : datetime
        Start of the acquisition window (inclusive).
    end_date : datetime
        End of the acquisition window (inclusive).

    Returns
    -------
    list[pystac.Item]
        Deduplicated STAC items (one per product), sorted by ID.
    """
    client = Client.open(EARTH_SEARCH_URL)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=(
            f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
            f"{end_date.strftime('%Y-%m-%dT23:59:59Z')}"
        ),
        max_items=10_000,
    )

    # Deduplicate by item ID
    seen: set[str] = set()
    unique = []
    for item in search.items():
        if item.id not in seen:
            seen.add(item.id)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _boa_offset(item: Any) -> int:
    """Return the BOA additive offset inferred from the processing baseline.

    Processing baseline >= 04.00 (introduced November 2022) applies a uniform
    offset of -1000 to all bands before scaling.  Earlier baselines use 0.

    Parameters
    ----------
    item : pystac.Item
        STAC item.

    Returns
    -------
    int
        Additive integer offset applied as ``(raw_dn + offset) / 10_000``.

    References
    ----------
    .. [ESA2022] ESA, Sentinel-2 Processing Baseline 04.00, Nov 2022.
       https://sentinels.copernicus.eu/web/sentinel/technical-guides/
       sentinel-2-msi/processing-baseline
    """
    # Earth Search COGs store reflectance-scale values with the offset already
    # applied; do not subtract -1000 a second time.
    if item.properties.get("earthsearch:boa_offset_applied"):
        return 0

    baseline_str = str(
        item.properties.get("processing:baseline")
        or item.properties.get("s2:processing_baseline")
        or "0"
    )
    try:
        baseline = float(baseline_str)
    except ValueError:
        baseline = 0.0
    return -1000 if baseline >= 4.0 else 0


def _window_for_bbox(
    src: rasterio.io.DatasetReader,
    bbox_4326: list[float],
) -> Window | None:
    """Return the rasterio Window for a EPSG:4326 bbox clipped to src bounds.

    Returns None when the bbox does not overlap the raster.
    """
    bbox_geom = {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox_4326[0], bbox_4326[1]],
                [bbox_4326[2], bbox_4326[1]],
                [bbox_4326[2], bbox_4326[3]],
                [bbox_4326[0], bbox_4326[3]],
                [bbox_4326[0], bbox_4326[1]],
            ]
        ],
    }
    src_geom = transform_geom("EPSG:4326", src.crs.to_wkt(), bbox_geom)
    xs = [c[0] for c in src_geom["coordinates"][0][:4]]
    ys = [c[1] for c in src_geom["coordinates"][0][:4]]
    win = window_from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform)

    # Clamp to valid pixel extent
    col_off = win.col_off
    row_off = win.row_off
    col_end = col_off + win.width
    row_end = row_off + win.height

    col_off = max(0.0, col_off)
    row_off = max(0.0, row_off)
    col_end = min(float(src.width), col_end)
    row_end = min(float(src.height), row_end)

    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _read_window(
    href: str,
    bbox_4326: list[float],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Range-read a single-band array for the bbox window from a COG URL.

    Returns None when the bbox does not overlap the raster.
    """
    with rasterio.open(href) as src:
        win = _window_for_bbox(src, bbox_4326)
        if win is None:
            return None
        data = src.read(1, window=win)
        profile: dict[str, Any] = {
            "crs": src.crs,
            "transform": window_transform(win, src.transform),
            "height": data.shape[0],
            "width": data.shape[1],
            "dtype": src.dtypes[0],
        }
    return data, profile


def _resample_to_shape(
    data: np.ndarray,
    target_shape: tuple[int, int],
    order: int,
) -> np.ndarray:
    """Resize a 2-D array to ``target_shape`` via ``scipy.ndimage.zoom``."""
    if data.shape == target_shape:
        return data
    zy = target_shape[0] / data.shape[0]
    zx = target_shape[1] / data.shape[1]
    return zoom(data, (zy, zx), order=order)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_gcc_s2(s2_dir: Path, output_dir: Path) -> None:
    """Compute GCC from cloud-native S2 REFL files and write single-band GeoTIFFs.

    Reads every ``*_REFL.tif`` produced by :func:`download_s2_window` (band
    order B02/B03/B04) and writes a co-located ``*_GCC.tif``.

    Parameters
    ----------
    s2_dir : Path
        Directory containing ``*_REFL.tif`` files.
    output_dir : Path
        Destination for ``*_GCC.tif`` files (created if absent).

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for src_path in s2_dir.glob("*_REFL.tif"):
        with rasterio.open(src_path) as src:
            b, g, r = src.read(1), src.read(2), src.read(3)
            profile = src.profile
        gcc = g / (b + g + r + 1e-10)
        gcc[b + g + r == 0] = 0  # propagate cloud mask
        profile.update(count=1)
        out = output_dir / src_path.name.replace("_REFL.tif", "_GCC.tif")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(gcc[np.newaxis].astype("float32"))


def download_s2_window(
    items: list[Any],
    bbox: list[float],
    output_dir: Path,
    bands: list[str],
) -> None:
    """Range-read S2 L2A COG windows and write masked REFL GeoTIFFs.

    For each STAC item the function:

    1. Issues HTTP range requests (via GDAL/rasterio) directly to the COG
       assets on AWS — no full-file download.
    2. Range-reads the SCL COG for pixel-level cloud masking.
    3. Infers the BOA offset from ``processing:baseline`` (−1000 for
       baseline ≥ 04.00, 0 otherwise) and scales to surface reflectance
       via ``(raw_dn + offset) / 10_000``.
    4. Masks cloud/shadow pixels to 0 (SCL classes 0, 3, >7), matching the
       convention of ``s2_processing.extract_mask_s2_bands``.
    5. Writes ``{item.id}_REFL.tif`` to ``output_dir``.

    Output filenames mirror the S2 product naming convention
    (e.g. ``S2A_MSIL2A_20230911T114111_…_REFL.tif``) so that
    ``efast.fusion(date_position=2)`` resolves acquisition dates without
    modification.

    Parameters
    ----------
    items : list[pystac.Item]
        STAC items from ``stac_search_s2``.
    bbox : list[float]
        ``[west, south, east, north]`` in EPSG:4326 — the AOI window to crop.
    output_dir : Path
        Destination for ``*_REFL.tif`` files (created if absent).
    bands : list[str]
        S2 band names, e.g. ``["B02", "B03", "B04", "B8A"]``.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(items, unit="granule", desc="S2 COG window read"):
        out_path = output_dir / f"{item.id}_REFL.tif"
        if out_path.is_file():
            continue

        offset = _boa_offset(item)

        # --- Per-band COG window read ---
        band_arrays: list[np.ndarray] = []
        ref_profile: dict[str, Any] | None = None
        skip = False

        for band_name in bands:
            asset_key = _BAND_ASSETS.get(band_name)
            if asset_key is None or asset_key not in item.assets:
                skip = True
                break
            href = item.assets[asset_key].href
            result = _read_window(href, bbox)
            if result is None:
                skip = True
                break
            data, profile = result
            if ref_profile is None:
                ref_profile = profile
            else:
                target_shape = (ref_profile["height"], ref_profile["width"])
                if data.shape != target_shape:
                    # Mixed native resolutions (e.g. 10 m B02 vs 20 m B8A)
                    data = _resample_to_shape(data, target_shape, order=1)
            band_arrays.append(data.astype("float32"))

        if skip or ref_profile is None:
            continue

        # --- SCL COG window read for cloud mask ---
        target_shape = (ref_profile["height"], ref_profile["width"])
        scl_asset = item.assets.get(_SCL_ASSET)

        if scl_asset is None:
            cloud_mask = np.zeros(target_shape, dtype=bool)
        else:
            scl_result = _read_window(scl_asset.href, bbox)
            if scl_result is None:
                cloud_mask = np.zeros(target_shape, dtype=bool)
            else:
                scl_data, _ = scl_result
                # Resize SCL to match band dimensions when resolutions differ
                # (e.g. 20 m SCL vs 10 m band) using nearest-neighbour
                if scl_data.shape != target_shape:
                    scl_data = _resample_to_shape(scl_data, target_shape, order=0)
                    scl_data = scl_data.astype(np.uint8)
                cloud_mask = (scl_data == 0) | (scl_data == 3) | (scl_data > 7)

        # --- Apply BOA offset, scale, mask, and write ---
        stacked = np.stack(band_arrays)  # (n_bands, H, W)
        stacked = (stacked + offset) / 10_000.0
        np.clip(stacked, 0, None, out=stacked)
        stacked[:, cloud_mask] = 0.0

        # Pad to a multiple of the EFAST S2/S3 ratio so distance_to_cloud and
        # fusion upsampling use matching HR/LR grid sizes.
        pad_h = (_EFAST_RATIO_ALIGN - stacked.shape[1] % _EFAST_RATIO_ALIGN) % _EFAST_RATIO_ALIGN
        pad_w = (_EFAST_RATIO_ALIGN - stacked.shape[2] % _EFAST_RATIO_ALIGN) % _EFAST_RATIO_ALIGN
        if pad_h or pad_w:
            stacked = np.pad(
                stacked,
                ((0, 0), (0, pad_h), (0, pad_w)),
                constant_values=0,
            )

        out_profile = {
            "driver": "GTiff",
            "count": len(bands),
            "dtype": "float32",
            "nodata": 0,
            "crs": ref_profile["crs"],
            "transform": ref_profile["transform"],
            "height": stacked.shape[1],
            "width": stacked.shape[2],
            "compress": "lzw",
        }
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(stacked)
            for i, band_name in enumerate(bands, 1):
                dst.set_band_description(i, band_name)
