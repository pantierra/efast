# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024 DHI A/S & contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: pantierra
"""

"""Cloud-native variant of run_efast.py.

Both Sentinel-2 and Sentinel-3 acquisitions are handled without local SNAP
or full-granule downloads:

* **Sentinel-2** — streams per-band COG windows from the Element84 Earth
  Search catalogue (AWS) using HTTP range reads; no authentication required.
* **Sentinel-3** — performs server-side spatial subsetting via CDSE OpenEO,
  downloading a small NetCDF for the AOI only.  SNAP is not required.

All downstream processing steps (distance-to-clouds, S3 compositing,
smoothing, reprojection, EFAST fusion) are unchanged from run_efast.py.
"""

import argparse

from datetime import datetime, timedelta
from pathlib import Path

from dateutil import rrule

import efast.efast as efast
import efast.s2_processing as s2
import efast.s3_processing as s3

from efast.s2_cloud_native import compute_gcc_s2, download_s2_window, stac_search_s2, wkt_to_bbox
from efast.s3_openeo import compute_gcc_s3, download_s3_openeo
from run_efast import CREDENTIALS

TEST_DATA_ROOT = Path("./test_data").absolute()


def output_dirs(year: str, site_name: str) -> dict[str, Path]:
    """Return pipeline output directories under ``test_data/year/sitename/``."""
    path = TEST_DATA_ROOT / year / site_name
    return {
        "s3_binning_dir": path / "S3/binning",
        "s3_composites_dir": path / "S3/composites",
        "s3_blured_dir": path / "S3/blurred",
        "s3_calibrated_dir": path / "S3/calibrated",
        "s3_reprojected_dir": path / "S3/reprojected",
        "s3_gcc_dir": path / "S3/gcc",
        "s2_processed_dir": path / "S2/processed",
        "fusion_dir": path / "fusion_results",
    }


def main(
    start_date: str,
    end_date: str,
    aoi_geometry: str,
    s2_bands: list,
    mosaic_days: int,
    step: int,
    cdse_credentials: dict,
    ratio: int,
    site_name: str,
):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    dirs = output_dirs(str(start_date.year), site_name)
    s3_binning_dir = dirs["s3_binning_dir"]
    s3_composites_dir = dirs["s3_composites_dir"]
    s3_blured_dir = dirs["s3_blured_dir"]
    s3_calibrated_dir = dirs["s3_calibrated_dir"]
    s3_reprojected_dir = dirs["s3_reprojected_dir"]
    s3_gcc_dir = dirs["s3_gcc_dir"]
    s2_processed_dir = dirs["s2_processed_dir"]
    fusion_dir = dirs["fusion_dir"]

    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Sentinel-3: server-side AOI subsetting via CDSE OpenEO (no SNAP required)
    download_s3_openeo(
        start_date,
        end_date,
        aoi_geometry,
        s3_binning_dir,
        cdse_credentials,
    )

    # Sentinel-2: COG window reads from AWS Earth Search (no auth required)
    download_s2_cloud_native(
        start_date,
        end_date,
        aoi_geometry,
        s2_processed_dir,
        s2_bands,
    )

    s2.distance_to_clouds(s2_processed_dir, ratio=ratio)
    compute_gcc_s2(s2_processed_dir, s2_processed_dir)

    s3.produce_median_composite(
        s3_binning_dir,
        s3_composites_dir,
        mosaic_days=mosaic_days,
        step=step,
        s3_bands=None,
    )
    s3.smoothing(s3_composites_dir, s3_blured_dir, std=1, preserve_nan=False)
    s3.reformat_s3(
        s3_blured_dir,
        s3_calibrated_dir,
        scaling_factor=s3.SDR_DN_TO_REFLECTANCE,
    )
    s3.reproject_and_crop_s3(s3_calibrated_dir, s2_processed_dir, s3_reprojected_dir)
    compute_gcc_s3(s3_reprojected_dir, s3_gcc_dir)

    for date in rrule.rrule(
        rrule.DAILY,
        dtstart=start_date + timedelta(step),
        until=end_date - timedelta(step),
        interval=step,
    ):
        efast.fusion(
            date,
            s3_gcc_dir,
            s2_processed_dir,
            fusion_dir,
            product="GCC",
            ratio=ratio,
            max_days=100,
            minimum_acquisition_importance=0,
        )


def download_s2_cloud_native(
    start_date,
    end_date,
    aoi_geometry,
    s2_processed_dir,
    s2_bands,
):
    """Range-read S2 L2A COG windows and write masked REFL GeoTIFFs.

    Replaces the ``download_list_safe`` + ``extract_mask_s2_bands`` pair from
    run_efast.py.  Output files are written directly to ``s2_processed_dir``
    in the same ``*_REFL.tif`` format so that ``s2.distance_to_clouds`` and
    ``efast.fusion`` can be used unchanged.

    Uses the AWS Element84 Earth Search catalogue, which hosts S2 L2A as
    Cloud-Optimised GeoTIFFs.  No authentication is required and only the
    bytes covering the AOI are transferred.

    Parameters
    ----------
    start_date : datetime
    end_date : datetime
    aoi_geometry : str
        WKT geometry string (POINT or polygon in EPSG:4326).
    s2_processed_dir : Path
        Destination directory for ``*_REFL.tif`` files.
    s2_bands : list[str]
        Band names to download, e.g. ``["B02", "B03", "B04", "B8A"]``.

    Returns
    -------
    None
    """
    bbox = wkt_to_bbox(aoi_geometry)
    items = stac_search_s2(bbox, start_date, end_date)
    download_s2_window(items, bbox, s2_processed_dir, bands=s2_bands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Cloud-native EFAST: S2 via COG range reads, S3 via CDSE OpenEO. "
            "No SNAP required."
        )
    )
    parser.add_argument("--start-date", default="2023-09-11")
    parser.add_argument("--end-date", default="2023-09-21")
    parser.add_argument(
        "--aoi-geometry", default="POINT (-15.432283 15.402828)"
    )  # Dahra EC tower
    parser.add_argument("--s2-bands", nargs="+", default=["B02", "B03", "B04", "B8A"])
    parser.add_argument("--mosaic-days", type=int, default=100)
    parser.add_argument("--step", type=int, required=False, default=2)
    parser.add_argument("--cdse-credentials", default=CREDENTIALS)
    parser.add_argument("--ratio", required=False, type=int, default=30)
    parser.add_argument(
        "--site-name",
        default="dahra",
        help="Site label used in output path test_data/YEAR/SITE_NAME/...",
    )

    args = parser.parse_args()

    main(
        start_date=args.start_date,
        end_date=args.end_date,
        aoi_geometry=args.aoi_geometry,
        s2_bands=args.s2_bands,
        step=args.step,
        mosaic_days=args.mosaic_days,
        cdse_credentials=args.cdse_credentials,
        ratio=args.ratio,
        site_name=args.site_name,
    )
