# -*- coding: utf-8 -*-
"""
MIT License

Sentinel-3 SYN L2 acquisition via CDSE OpenEO.

Downloads SENTINEL3_SYN_L2_SYN (SY_2_SYN) for an AOI bounding box via
OpenEO server-side spatial subsetting. Only the pixels covering the AOI
are transferred; no full granule download or SNAP binning is required.

The output per-date GeoTIFFs are named to match the pattern expected by
``s3_processing.produce_median_composite``:

    S3_{YYYYMMDD}_{n}__{YYYYMMDD}T120000.tif

Key functions
-------------
download_s3_openeo : download, split and write per-date S3 GeoTIFFs
"""

from __future__ import annotations

import time

from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import openeo
import rasterio
import requests

from rasterio.transform import from_bounds

from efast.s2_cloud_native import wkt_to_bbox

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
OPENEO_URL = "openeo.dataspace.copernicus.eu"
COLLECTION = "SENTINEL3_SYN_L2_SYN"

# SYN L2 surface directional reflectance bands used by EFAST
# (mirrors run_efast.py --s3-bands default and processing/acquisition_s3l2.py)
_OPENEO_BANDS = (
    "Syn_Oa04_reflectance",
    "Syn_Oa06_reflectance",
    "Syn_Oa08_reflectance",
    "Syn_Oa17_reflectance",
)
_BAND_NAMES = ("SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17")


def _cdse_token(username: str, password: str) -> str:
    """Obtain a CDSE bearer token via password grant."""
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": "cdse-public",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _netcdf_to_geotiffs(nc_path: Path, output_dir: Path) -> int:
    """Split an OpenEO NetCDF into per-date GeoTIFFs.

    Output filenames match the ``S3*__YYYYMMDDTHHMMSS.tif`` pattern that
    ``s3_processing.produce_median_composite`` expects.

    Parameters
    ----------
    nc_path : Path
        Path to the downloaded NetCDF file.
    output_dir : Path
        Directory where per-date GeoTIFFs are written.

    Returns
    -------
    int
        Number of GeoTIFFs written.
    """
    written = 0
    with netCDF4.Dataset(str(nc_path), "r") as nc:
        times = netCDF4.num2date(nc.variables["t"][:], nc.variables["t"].units)
        x_coords = nc.variables["x"][:]
        y_coords = nc.variables["y"][:]
        transform = from_bounds(
            float(x_coords.min()),
            float(y_coords.min()),
            float(x_coords.max()),
            float(y_coords.max()),
            len(x_coords),
            len(y_coords),
        )

        date_counts: dict[str, int] = {}
        for t_idx, time_val in enumerate(times):
            dt = (
                time_val
                if isinstance(time_val, datetime)
                else netCDF4.num2date(nc.variables["t"][t_idx], nc.variables["t"].units)
            )
            date_str = dt.strftime("%Y%m%d")
            n = date_counts.get(date_str, 0)
            date_counts[date_str] = n + 1

            stacked = np.stack(
                [nc.variables[b][t_idx, :, :] for b in _OPENEO_BANDS], axis=0
            )
            # Filename satisfies both the S3*.tif glob and the
            # __YYYYMMDDTHHMMSS regex in produce_median_composite
            filename = f"S3_{date_str}_{n}__{date_str}T120000.tif"
            with rasterio.open(
                output_dir / filename,
                "w",
                driver="GTiff",
                height=len(y_coords),
                width=len(x_coords),
                count=len(_OPENEO_BANDS),
                dtype=stacked.dtype,
                crs="EPSG:32632",
                transform=transform,
                compress="lzw",
            ) as dst:
                dst.write(stacked)
                for i, band_name in enumerate(_BAND_NAMES, 1):
                    dst.set_band_description(i, band_name)
            written += 1

    return written


def compute_gcc_s3(s3_dir: Path, output_dir: Path) -> None:
    """Compute GCC from reprojected S3 composites and write single-band GeoTIFFs.

    Reads every ``composite_*.tif`` (band order Oa04/Oa06/Oa08/Oa17 per
    :data:`_BAND_NAMES`) and writes a single-band GCC ``composite_*.tif``.

    Parameters
    ----------
    s3_dir : Path
        Directory containing ``composite_*.tif`` files.
    output_dir : Path
        Destination for single-band GCC composites (created if absent).

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for src_path in s3_dir.glob("composite_*.tif"):
        with rasterio.open(src_path) as src:
            b, g, r = src.read(1), src.read(2), src.read(3)
            profile = src.profile
        gcc = g / (b + g + r + 1e-10)
        gcc[np.isnan(b) | np.isnan(g) | np.isnan(r)] = np.nan
        profile.update(count=1, dtype="float32")
        with rasterio.open(output_dir / src_path.name, "w", **profile) as dst:
            dst.write(gcc[np.newaxis].astype("float32"))


def download_s3_openeo(
    start_date: datetime,
    end_date: datetime,
    aoi_geometry: str,
    output_dir: Path,
    credentials: dict[str, str],
) -> None:
    """Download S3 SYN L2 SDR for an AOI via CDSE OpenEO, server-side clipped.

    Replaces ``download_s3_from_cdse`` + ``s3.binning_s3`` from run_efast.py.
    OpenEO handles spatial subsetting and reprojection to EPSG:32632
    server-side, so only the AOI pixels are transferred and no SNAP
    installation is required.

    Output GeoTIFFs are named ``S3_{YYYYMMDD}_{n}__{YYYYMMDD}T120000.tif``
    and placed in ``output_dir`` so that ``s3_processing.produce_median_composite``
    can consume them directly.

    Parameters
    ----------
    start_date : datetime
        Start of the acquisition window (inclusive).
    end_date : datetime
        End of the acquisition window (inclusive).
    aoi_geometry : str
        WKT geometry (POINT or polygon, EPSG:4326) defining the AOI.
    output_dir : Path
        Destination for per-date GeoTIFFs (created if absent).
    credentials : dict
        Dict with ``username`` and ``password`` keys (CDSE account).

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if list(output_dir.glob("S3*.tif")):
        print("[S3-OEO] Skipping — output_dir already contains S3 GeoTIFFs")
        return

    bbox = wkt_to_bbox(aoi_geometry)
    spatial_extent = {
        "west": bbox[0],
        "east": bbox[2],
        "south": bbox[1],
        "north": bbox[3],
    }

    print("[S3-OEO] Authenticating with CDSE...")
    token = _cdse_token(credentials["username"], credentials["password"])
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc_access_token(token)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    print(f"[S3-OEO] Loading {COLLECTION} ({start_str} → {end_str})...")
    datacube = conn.load_collection(
        COLLECTION,
        spatial_extent=spatial_extent,
        temporal_extent=[start_str, end_str],
        bands=list(_OPENEO_BANDS),
    ).resample_spatial(projection=32632)

    nc_path = output_dir / "_s3_syn_l2.nc"
    print(f"[S3-OEO] Downloading NetCDF to {nc_path}...")
    t0 = time.time()
    datacube.download(str(nc_path), format="NetCDF")
    print(f"[S3-OEO] Download completed in {time.time() - t0:.1f}s")

    print("[S3-OEO] Splitting into per-date GeoTIFFs...")
    written = _netcdf_to_geotiffs(nc_path, output_dir)
    nc_path.unlink(missing_ok=True)
    print(f"[S3-OEO] {written} GeoTIFFs written to {output_dir}")
