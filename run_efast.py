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

@author: rmgu, pase
"""

import argparse
import logging
import os
import time
import zipfile

from datetime import datetime, timedelta
from pathlib import Path

from creodias_finder import query
from creodias_finder.download import _get_token, download
from dateutil import rrule
from pystac_client import Client
from shapely import wkt
from tqdm import tqdm

import efast.efast as efast
import efast.s2_processing as s2
import efast.s3_processing as s3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# CDSE credentials to download Sentinel-2 and Sentinel-3 imagery
def get_credentials_from_env():
    """
    Read CDSE credentials from the environment variables CDSE_USER and CDSE_PASSWORD.

    Returns
    -------
    Dictionary containing "username" and "password" keys mapped to the values of the CDSE_USER and CDSE_PASSWORD
    environment variables.

    """
    username = os.getenv("CDSE_USER")
    password = os.getenv("CDSE_PASSWORD")

    return {
        "username": username,
        "password": password
    }


CREDENTIALS = get_credentials_from_env()


# Test parameters
path = Path("./test_data").absolute()
s3_download_dir = path / "S3/raw"
s3_binning_dir = path / "S3/binning"
s3_composites_dir = path / "S3/composites"
s3_blured_dir = path / "S3/blurred"
s3_calibrated_dir = path / "S3/calibrated"
s3_reprojected_dir = path / "S3/reprojected"
s2_download_dir = path / "S2/raw"
s2_processed_dir = path / "S2/processed"
fusion_dir = path / "fusion_results"


def main(
    start_date: str,
    end_date: str,
    aoi_geometry: str,
    s3_sensor: str,
    s3_bands: list,
    s2_bands: list,
    mosaic_days: int,
    step: int,
    cdse_credentials: dict,
    ratio: int,
    snap_gpt_path: str = "gpt",
    log_level: str = "INFO",
    data_source: str = "cdse",
):
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("Starting EFAST Processing")
    logger.info("=" * 70)
    
    # Transform parameters
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if s3_sensor == "SYN":
        instrument = "SYNERGY"
    else:
        instrument = s3_sensor

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"AOI Geometry: {aoi_geometry}")
    logger.info(f"S3 Sensor: {s3_sensor} (instrument: {instrument})")
    logger.info(f"S3 Bands: {s3_bands}")
    logger.info(f"S2 Bands: {s2_bands}")
    logger.info(f"Mosaic days: {mosaic_days}, Step: {step} days, Ratio: {ratio}")
    logger.info(f"Data source: {data_source}")

    # Create directories if necessary
    logger.info("Creating output directories...")
    for folder in [
        s3_download_dir,
        s3_binning_dir,
        s3_composites_dir,
        s3_blured_dir,
        s3_calibrated_dir,
        s3_reprojected_dir,
        s2_processed_dir,
        s2_download_dir,
        fusion_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created/verified directory: {folder}")

    # Download or query data
    logger.info("=" * 70)
    if data_source == "stac":
        logger.info("STEP 1: Querying data from STAC API")
    else:
        logger.info("STEP 1: Downloading data from CDSE")
    logger.info("=" * 70)
    download_start = time.time()
    s2_stac_items, s3_stac_items = download_data(
        start_date,
        end_date,
        aoi_geometry,
        s2_download_dir,
        s3_download_dir,
        cdse_credentials,
        data_source=data_source)
    download_time = time.time() - download_start
    logger.info(f"Data query/download completed in {download_time:.1f} seconds")
    
    if data_source == "stac":
        logger.info(f"Found {len(s2_stac_items) if s2_stac_items else 0} S2 STAC items, {len(s3_stac_items) if s3_stac_items else 0} S3 STAC items")
    else:
        # Check downloaded files
        s3_zips = list(s3_download_dir.glob("*.zip"))
        s2_zips = list(s2_download_dir.glob("*.zip"))
        logger.info(f"Downloaded files: {len(s3_zips)} S3 zip files, {len(s2_zips)} S2 zip files")

    # Sentinel-2 pre-processing
    logger.info("=" * 70)
    logger.info("STEP 2: Sentinel-2 Pre-processing")
    logger.info("=" * 70)
    s2_start = time.time()
    
    # Check if we have data to process
    if data_source == "stac" and (not s2_stac_items or len(s2_stac_items) == 0):
        logger.error("No Sentinel-2 STAC items found. Cannot proceed with processing.")
        logger.error("Please check:")
        logger.error("  1. Date range has available data")
        logger.error("  2. AOI geometry is correct")
        logger.error("  3. STAC API is accessible")
        logger.error("Consider using --data-source cdse as an alternative")
        raise ValueError("No Sentinel-2 data available for processing")
    
    logger.info("Extracting and masking Sentinel-2 bands...")
    s2.extract_mask_s2_bands(
        s2_download_dir,
        s2_processed_dir,
        bands=s2_bands,
        stac_items=s2_stac_items if data_source == "stac" else None,
    )
    s2_files = list(s2_processed_dir.glob("*REFL.tif"))
    logger.info(f"Extracted {len(s2_files)} Sentinel-2 reflectance files")
    
    if len(s2_files) == 0:
        logger.error("No Sentinel-2 files were processed. Cannot continue.")
        raise ValueError("No Sentinel-2 files available for further processing")
    
    logger.info("Computing distance to clouds...")
    s2.distance_to_clouds(
        s2_processed_dir,
        ratio=ratio,
    )
    
    logger.info("Getting Sentinel-2 footprint...")
    footprint = s2.get_wkt_footprint(
        s2_processed_dir,
    )
    logger.debug(f"Footprint: {footprint[:100]}...")
    s2_time = time.time() - s2_start
    logger.info(f"Sentinel-2 pre-processing completed in {s2_time:.1f} seconds")

    # Sentinel-3 pre-processing
    logger.info("=" * 70)
    logger.info("STEP 3: Sentinel-3 Pre-processing")
    logger.info("=" * 70)
    s3_start = time.time()
    
    logger.info("Binning Sentinel-3 data...")
    s3.binning_s3(
        s3_download_dir,
        s3_binning_dir,
        footprint=footprint,
        s3_bands=s3_bands,
        instrument=instrument,
        aggregator="mean",
        snap_gpt_path=snap_gpt_path,
        snap_memory="24G",
        snap_parallelization=1,
        stac_items=s3_stac_items if data_source == "stac" else None,
    )
    s3_binned = list(s3_binning_dir.glob("*.tif"))
    logger.info(f"Binned {len(s3_binned)} Sentinel-3 files")
    
    logger.info("Producing median composites...")
    s3.produce_median_composite(
        s3_binning_dir,
        s3_composites_dir,
        mosaic_days=mosaic_days,
        step=step,
        s3_bands=None,
    )
    s3_composites = list(s3_composites_dir.glob("composite*.tif"))
    logger.info(f"Created {len(s3_composites)} median composites")
    
    logger.info("Smoothing composites...")
    s3.smoothing(
        s3_composites_dir,
        s3_blured_dir,
        std=1,
        preserve_nan=False,
    )
    
    logger.info("Reformatting Sentinel-3 data...")
    s3.reformat_s3(
        s3_blured_dir,
        s3_calibrated_dir,
    )
    
    logger.info("Reprojecting and cropping Sentinel-3 data...")
    s3.reproject_and_crop_s3(
        s3_calibrated_dir,
        s2_processed_dir,
        s3_reprojected_dir,
    )
    s3_reprojected = list(s3_reprojected_dir.glob("composite*.tif"))
    logger.info(f"Reprojected {len(s3_reprojected)} Sentinel-3 files")
    s3_time = time.time() - s3_start
    logger.info(f"Sentinel-3 pre-processing completed in {s3_time:.1f} seconds")

    # Perform EFAST fusion
    logger.info("=" * 70)
    logger.info("STEP 4: Performing EFAST Fusion")
    logger.info("=" * 70)
    fusion_start = time.time()
    
    # Calculate number of dates
    dates = list(rrule.rrule(
        rrule.DAILY,
        dtstart=start_date + timedelta(step),
        until=end_date - timedelta(step),
        interval=step,
    ))
    logger.info(f"Processing {len(dates)} dates for fusion...")
    
    for i, date in enumerate(dates, 1):
        logger.info(f"Processing date {i}/{len(dates)}: {date.date()}")
        efast.fusion(
            date,
            s3_reprojected_dir,
            s2_processed_dir,
            fusion_dir,
            product="REFL",
            ratio=ratio,
            max_days=100,
            minimum_acquisition_importance=0,
        )
    
    fusion_files = list(fusion_dir.glob("REFL_*.tif"))
    logger.info(f"Created {len(fusion_files)} fused output files")
    fusion_time = time.time() - fusion_start
    logger.info(f"EFAST fusion completed in {fusion_time:.1f} seconds")
    
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("EFAST Processing Summary")
    logger.info("=" * 70)
    logger.info(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"  - Download: {download_time:.1f}s ({download_time/total_time*100:.1f}%)")
    logger.info(f"  - S2 Pre-processing: {s2_time:.1f}s ({s2_time/total_time*100:.1f}%)")
    logger.info(f"  - S3 Pre-processing: {s3_time:.1f}s ({s3_time/total_time*100:.1f}%)")
    logger.info(f"  - Fusion: {fusion_time:.1f}s ({fusion_time/total_time*100:.1f}%)")
    logger.info(f"Output directory: {fusion_dir}")
    logger.info("=" * 70)


def wkt_to_bbox(wkt_geometry):
    """
    Convert WKT geometry string to bbox [minx, miny, maxx, maxy] for STAC queries.
    
    Parameters
    ----------
    wkt_geometry : str
        WKT geometry string (e.g., "POINT (lon lat)" or "POLYGON (...)")
    
    Returns
    -------
    list
        Bounding box as [minx, miny, maxx, maxy]
    """
    geom = wkt.loads(wkt_geometry)
    bounds = geom.bounds
    # For POINT geometries, create a buffer to get a bbox
    # Sentinel-2 tiles are ~110km, so use ~0.5 degree buffer (~55km) to ensure we get tiles
    if geom.geom_type == 'Point':
        # Create a buffer around the point (0.5 degrees ~55km)
        buffered = geom.buffer(0.5)
        bounds = buffered.bounds
    return [bounds[0], bounds[1], bounds[2], bounds[3]]


def query_stac_s2(start_date, end_date, aoi_geometry):
    """
    Query AWS Element84 STAC API for Sentinel-2 L2A COGs.
    
    Parameters
    ----------
    start_date : datetime
        Start date for the query
    end_date : datetime
        End date for the query
    aoi_geometry : str
        WKT geometry string for area of interest
    
    Returns
    -------
    list
        List of STAC items
    """
    # Use v1 endpoint (more compliant)
    stac_endpoint = "https://earth-search.aws.element84.com/v1"
    collection = "sentinel-2-l2a"  # Correct collection name for Element84
    
    # Convert WKT to bbox
    bbox = wkt_to_bbox(aoi_geometry)
    
    # Format datetime for STAC query
    datetime_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    logger.info(f"Querying STAC API: {stac_endpoint}")
    logger.info(f"Collection: {collection}")
    logger.info(f"Date range: {datetime_range}")
    logger.info(f"Bbox: {bbox}")
    
    try:
        catalog = Client.open(stac_endpoint)
        
        # For large date ranges, query in monthly chunks to avoid API timeouts
        days_diff = (end_date - start_date).days
        if days_diff > 90:
            logger.info(f"Large date range ({days_diff} days), querying in monthly chunks...")
            all_items = []
            current_start = start_date
            while current_start < end_date:
                # Calculate end of current month or end_date, whichever comes first
                if current_start.month == 12:
                    current_end = datetime(current_start.year + 1, 1, 1)
                else:
                    current_end = datetime(current_start.year, current_start.month + 1, 1)
                current_end = min(current_end, end_date)
                
                chunk_range = f"{current_start.strftime('%Y-%m-%d')}/{current_end.strftime('%Y-%m-%d')}"
                logger.debug(f"Querying chunk: {chunk_range}")
                
                search = catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    datetime=chunk_range,
                    limit=1000,
                )
                chunk_items = list(search.items())
                all_items.extend(chunk_items)
                logger.debug(f"Found {len(chunk_items)} items in chunk")
                
                current_start = current_end
            
            # Remove duplicates based on item ID
            seen_ids = set()
            unique_items = []
            for item in all_items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    unique_items.append(item)
            
            logger.info(f"Found {len(unique_items)} unique Sentinel-2 STAC items")
            return unique_items
        else:
            # For smaller date ranges, query directly
            search = catalog.search(
                collections=[collection],
                bbox=bbox,
                datetime=datetime_range,
                limit=1000,
            )
            items = list(search.items())
            logger.info(f"Found {len(items)} Sentinel-2 STAC items")
            return items
    except Exception as e:
        logger.error(f"Error querying STAC API for Sentinel-2: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        logger.warning("STAC query failed. You may need to use --data-source cdse instead")
        return []


def query_stac_s3(start_date, end_date, aoi_geometry):
    """
    Query MEEO STAC Catalog for Sentinel-3 COGs.
    
    Note: MEEO provides a static STAC catalog, not a STAC API, so we need to
    load the catalog and search through it manually. For now, we'll return
    an empty list and fall back to CDSE, as the catalog structure needs
    further investigation.
    
    Parameters
    ----------
    start_date : datetime
        Start date for the query
    end_date : datetime
        End date for the query
    aoi_geometry : str
        WKT geometry string for area of interest
    
    Returns
    -------
    list
        List of STAC items, or empty list if query fails
    """
    stac_endpoint = "https://meeo-s3.s3.amazonaws.com/catalog.json"
    
    logger.info(f"Querying STAC Catalog: {stac_endpoint}")
    logger.info("Note: MEEO STAC catalog is static and requires manual traversal")
    logger.info("Sentinel-3 STAC support is not yet fully implemented")
    logger.warning("Falling back to CDSE for Sentinel-3")
    
    # TODO: Implement proper STAC catalog traversal for MEEO
    # The MEEO catalog is a static STAC catalog, not a STAC API
    # We would need to traverse the catalog structure to find matching items
    return []


def download_data(
        start_date,
        end_date,
        aoi_geometry,
        s2_download_dir,
        s3_download_dir,
        credentials,
        data_source="cdse"):
    """
    Download or query data from CDSE or STAC API.
    
    Parameters
    ----------
    start_date : datetime
        Start date for the query
    end_date : datetime
        End date for the query
    aoi_geometry : str
        WKT geometry string for area of interest
    s2_download_dir : Path
        Directory for Sentinel-2 downloads (used for CDSE mode)
    s3_download_dir : Path
        Directory for Sentinel-3 downloads (used for CDSE mode)
    credentials : dict
        CDSE credentials (used for CDSE mode)
    data_source : str
        Data source: "stac" or "cdse" (default: "cdse")
    
    Returns
    -------
    tuple
        (s2_stac_items, s3_stac_items) - Lists of STAC items, or (None, None) for CDSE mode
    """
    if data_source == "stac":
        # Query STAC API
        s2_stac_items = query_stac_s2(start_date, end_date, aoi_geometry)
        s3_stac_items = query_stac_s3(start_date, end_date, aoi_geometry)
        
        # For Sentinel-3, if STAC query fails or returns no items, fallback to CDSE
        if not s3_stac_items:
            logger.warning("No Sentinel-3 items found via STAC, falling back to CDSE for S3")
            # Only try CDSE if credentials are available
            if credentials.get("username") and credentials.get("password"):
                download_from_cdse_s3_only(start_date, end_date, aoi_geometry, s3_download_dir, credentials)
            else:
                logger.warning("CDSE credentials not available, skipping Sentinel-3 download")
            s3_stac_items = None
        
        return s2_stac_items, s3_stac_items
    else:
        # Original CDSE download
        download_from_cdse(start_date, end_date, aoi_geometry, s2_download_dir, s3_download_dir, credentials)
        return None, None


def download_from_cdse(
        start_date,
        end_date,
        aoi_geometry,
        s2_download_dir,
        s3_download_dir,
        credentials):

    logger.info("Querying Sentinel-3 SYN data...")
    results = query.query('Sentinel3',
                          start_date=start_date,
                          end_date=end_date,
                          geometry=aoi_geometry,
                          instrument="SYNERGY",
                          productType="SY_2_SYN___",
                          timeliness="NT")
    s3_count = len(results)
    logger.info(f"Found {s3_count} Sentinel-3 SYN products")
    
    if s3_count > 0:
        logger.info(f"Downloading {s3_count} Sentinel-3 files...")
        download_list_safe([result['id'] for result in results.values()],
                               outdir=s3_download_dir,
                               threads=3,
                               **credentials)
        logger.info("Extracting Sentinel-3 zip files...")
        zip_count = 0
        for zip_file in s3_download_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(s3_download_dir)
            zip_count += 1
        logger.info(f"Extracted {zip_count} Sentinel-3 zip files")
    else:
        logger.warning("No Sentinel-3 products found for the specified criteria")

    logger.info("Querying Sentinel-2 L2A data...")
    results = query.query(
        'Sentinel2',
        start_date=start_date,
        end_date=end_date,
        geometry=aoi_geometry,
        productType="L2A",
    )
    s2_count = len(results)
    logger.info(f"Found {s2_count} Sentinel-2 L2A products")
    
    if s2_count > 0:
        logger.info(f"Downloading {s2_count} Sentinel-2 files...")
        download_list_safe(
            [result['id'] for result in results.values()],
            outdir=s2_download_dir,
            threads=3,
            **credentials,
        )
        logger.info("Extracting Sentinel-2 zip files...")
        zip_count = 0
        for zip_file in s2_download_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(s2_download_dir)
            zip_count += 1
        logger.info(f"Extracted {zip_count} Sentinel-2 zip files")
    else:
        logger.warning("No Sentinel-2 products found for the specified criteria")


def download_from_cdse_s3_only(
        start_date,
        end_date,
        aoi_geometry,
        s3_download_dir,
        credentials):
    """Download only Sentinel-3 data from CDSE (used as fallback when STAC fails)."""
    logger.info("Querying Sentinel-3 SYN data from CDSE...")
    results = query.query('Sentinel3',
                          start_date=start_date,
                          end_date=end_date,
                          geometry=aoi_geometry,
                          instrument="SYNERGY",
                          productType="SY_2_SYN___",
                          timeliness="NT")
    s3_count = len(results)
    logger.info(f"Found {s3_count} Sentinel-3 SYN products")
    
    if s3_count > 0:
        logger.info(f"Downloading {s3_count} Sentinel-3 files...")
        download_list_safe([result['id'] for result in results.values()],
                               outdir=s3_download_dir,
                               threads=3,
                               **credentials)
        logger.info("Extracting Sentinel-3 zip files...")
        zip_count = 0
        for zip_file in s3_download_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(s3_download_dir)
            zip_count += 1
        logger.info(f"Extracted {zip_count} Sentinel-3 zip files")
    else:
        logger.warning("No Sentinel-3 products found for the specified criteria")


def download_list_safe(uids, username, password, outdir, threads=1, show_progress=True):
    if show_progress:
        pbar = tqdm(total=len(uids), unit="files")

    def _download(uid):
        token = _get_token(username, password)
        outfile = Path(outdir) / f"{uid}.zip"
        download(
            uid, username, password, outfile=outfile, show_progress=False, token=token
        )
        if show_progress:
            pbar.update(1)
        return uid, outfile

    paths = [_download(u) for u in uids]
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2023-09-11")
    parser.add_argument("--end-date", default="2023-09-21")
    parser.add_argument("--aoi-geometry", default="POINT (-15.432283 15.402828)")  # Dahra EC tower
    parser.add_argument("--s3-sensor", default="SYN")
    parser.add_argument(
        "--s3-bands", nargs="+", default=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]
    )
    parser.add_argument("--s2-bands", nargs="+", default=["B02", "B03", "B04", "B8A"])
    parser.add_argument("--mosaic-days", type=int, default=100)
    parser.add_argument("--step", type=int, required=False, default=2)
    parser.add_argument("--cdse-credentials", default=CREDENTIALS)
    parser.add_argument("--snap-gpt-path", required=False, default="gpt")
    parser.add_argument("--ratio", required=False, type=int, default=30)
    parser.add_argument("--log-level", required=False, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level (default: INFO)")
    parser.add_argument("--data-source", required=False, default="cdse",
                        choices=["cdse", "stac"],
                        help="Data source: 'cdse' for CDSE download or 'stac' for STAC API (default: cdse)")

    args = parser.parse_args()

    main(
        start_date=args.start_date,
        end_date=args.end_date,
        aoi_geometry=args.aoi_geometry,
        s3_sensor=args.s3_sensor,
        s3_bands=args.s3_bands,
        s2_bands=args.s2_bands,
        step=args.step,
        mosaic_days=args.mosaic_days,
        cdse_credentials=args.cdse_credentials,
        ratio=args.ratio,
        snap_gpt_path=args.snap_gpt_path,
        log_level=args.log_level,
        data_source=args.data_source,
    )
