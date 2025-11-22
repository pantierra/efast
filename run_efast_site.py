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
Wrapper script to run EFAST for a selected site and season from selected_sites.geojson
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from run_efast import main, get_credentials_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_geojson(geojson_path):
    """Load and parse the GeoJSON file."""
    with open(geojson_path, 'r') as f:
        return json.load(f)


def find_site_and_season(geojson_data, sitename, season_year):
    """
    Find a specific site and season in the GeoJSON data.
    
    Parameters
    ----------
    geojson_data : dict
        Parsed GeoJSON data
    sitename : str
        Name of the site to find
    season_year : str
        Year of the season to find
        
    Returns
    -------
    tuple
        (site_feature, season_data) or (None, None) if not found
    """
    for feature in geojson_data.get('features', []):
        props = feature.get('properties', {})
        if props.get('sitename') == sitename:
            seasons = props.get('seasons', {})
            if season_year in seasons:
                return feature, seasons[season_year]
            else:
                print(f"Warning: Season {season_year} not found for site {sitename}")
                print(f"Available seasons: {list(seasons.keys())}")
                return None, None
    
    print(f"Error: Site '{sitename}' not found in GeoJSON")
    print(f"Available sites:")
    for feature in geojson_data.get('features', []):
        props = feature.get('properties', {})
        print(f"  - {props.get('sitename')}")
    return None, None


def point_to_wkt(point_coords):
    """
    Convert GeoJSON Point coordinates to WKT POINT format.
    
    Parameters
    ----------
    point_coords : list
        [longitude, latitude] coordinates
        
    Returns
    -------
    str
        WKT POINT string in format "POINT (lon lat)"
    """
    lon, lat = point_coords
    return f"POINT ({lon} {lat})"


def list_available_sites(geojson_data, show_seasons=False):
    """
    List all available sites and optionally their seasons.
    
    Parameters
    ----------
    geojson_data : dict
        Parsed GeoJSON data
    show_seasons : bool
        If True, also show available seasons for each site
    """
    print("\nAvailable sites:")
    print("=" * 60)
    for feature in geojson_data.get('features', []):
        props = feature.get('properties', {})
        sitename = props.get('sitename', 'Unknown')
        description = props.get('description', '')
        coords = feature.get('geometry', {}).get('coordinates', [])
        ndvi_selected = props.get('ndvi_selected', False)
        
        print(f"\nSite: {sitename}")
        print(f"  Description: {description}")
        print(f"  Coordinates: {coords[0]:.6f}, {coords[1]:.6f}")
        print(f"  Site-level ndvi_selected: {ndvi_selected}")
        
        if show_seasons:
            seasons = props.get('seasons', {})
            print(f"  Available seasons: {list(seasons.keys())}")
            for year, season_data in seasons.items():
                selected = season_data.get('ndvi_selected', False)
                s2_scenes = season_data.get('sentinel2_scenes', 0)
                s3_scenes = season_data.get('sentinel3_scenes', 0)
                start_date = season_data.get('season_start_date', 'N/A')
                end_date = season_data.get('season_end_date', 'N/A')
                print(f"    {year}: {start_date} to {end_date} "
                      f"(S2: {s2_scenes}, S3: {s3_scenes}, selected: {selected})")


def validate_season_data(season_data, sitename, season_year):
    """
    Validate that the season has sufficient data for processing.
    
    Parameters
    ----------
    season_data : dict
        Season data from GeoJSON
    sitename : str
        Site name (for error messages)
    season_year : str
        Season year (for error messages)
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    s2_scenes = season_data.get('sentinel2_scenes', 0)
    s3_scenes = season_data.get('sentinel3_scenes', 0)
    
    if s2_scenes == 0:
        print(f"Warning: Site {sitename} season {season_year} has no Sentinel-2 scenes.")
        print("Processing may fail or produce no results.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    if s3_scenes == 0:
        print(f"Warning: Site {sitename} season {season_year} has no Sentinel-3 scenes.")
        print("Processing may fail or produce no results.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True


def main_wrapper(
    geojson_path,
    sitename,
    season_year,
    s3_sensor="SYN",
    s3_bands=None,
    s2_bands=None,
    mosaic_days=100,
    step=2,
    ratio=30,
    snap_gpt_path="gpt",
    list_sites=False,
    log_level="INFO",
    data_source="cdse",
):
    """
    Main wrapper function to run EFAST for a selected site and season.
    
    Parameters
    ----------
    geojson_path : str or Path
        Path to the selected_sites.geojson file
    sitename : str
        Name of the site to process
    season_year : str
        Year of the season to process
    s3_sensor : str
        Sentinel-3 sensor type (default: "SYN")
    s3_bands : list
        List of Sentinel-3 bands (default: None, uses defaults)
    s2_bands : list
        List of Sentinel-2 bands (default: None, uses defaults)
    mosaic_days : int
        Mosaic days parameter (default: 100)
    step : int
        Step size in days (default: 2)
    ratio : int
        Resolution ratio (default: 30)
    snap_gpt_path : str
        Path to SNAP GPT executable (default: "gpt")
    list_sites : bool
        If True, list all available sites and exit
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Load GeoJSON
    logger.info(f"Loading GeoJSON file: {geojson_path}")
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        logger.error(f"GeoJSON file not found: {geojson_path}")
        sys.exit(1)
    
    geojson_data = load_geojson(geojson_path)
    logger.debug(f"Loaded GeoJSON with {len(geojson_data.get('features', []))} features")
    
    # List sites if requested
    if list_sites:
        list_available_sites(geojson_data, show_seasons=True)
        return
    
    # Find site and season
    logger.info(f"Searching for site '{sitename}' and season '{season_year}'...")
    site_feature, season_data = find_site_and_season(geojson_data, sitename, season_year)
    
    if site_feature is None or season_data is None:
        logger.error("Could not find site and season combination.")
        print("\nUse --list-sites to see available options.")
        sys.exit(1)
    
    logger.info(f"Found site '{sitename}' with season '{season_year}'")
    
    # Validate season data
    if not validate_season_data(season_data, sitename, season_year):
        logger.warning("Aborted by user.")
        sys.exit(1)
    
    # Extract parameters
    start_date = season_data.get('season_start_date')
    end_date = season_data.get('season_end_date')
    
    if not start_date or not end_date:
        print(f"Error: Missing dates for site {sitename} season {season_year}")
        sys.exit(1)
    
    # Convert Point geometry to WKT
    geometry = site_feature.get('geometry', {})
    if geometry.get('type') != 'Point':
        print(f"Error: Expected Point geometry, got {geometry.get('type')}")
        sys.exit(1)
    
    coords = geometry.get('coordinates')
    aoi_geometry = point_to_wkt(coords)
    
    # Set default bands if not provided
    if s3_bands is None:
        s3_bands = ["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]
    if s2_bands is None:
        s2_bands = ["B02", "B03", "B04", "B8A"]
    
    # Get credentials
    cdse_credentials = get_credentials_from_env()
    
    # Print summary
    logger.info("=" * 70)
    logger.info("EFAST Processing Configuration")
    logger.info("=" * 70)
    logger.info(f"Site: {sitename}")
    logger.info(f"Season: {season_year}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Coordinates: {coords[0]:.6f}, {coords[1]:.6f}")
    logger.info(f"AOI Geometry: {aoi_geometry}")
    logger.info(f"S3 Sensor: {s3_sensor}")
    logger.info(f"S3 Bands: {s3_bands}")
    logger.info(f"S2 Bands: {s2_bands}")
    logger.info(f"Mosaic days: {mosaic_days}")
    logger.info(f"Step: {step} days")
    logger.info(f"Ratio: {ratio}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Data source: {data_source}")
    logger.info("=" * 70)
    
    # Run EFAST
    try:
        logger.info("Starting EFAST main processing...")
        main(
            start_date=start_date,
            end_date=end_date,
            aoi_geometry=aoi_geometry,
            s3_sensor=s3_sensor,
            s3_bands=s3_bands,
            s2_bands=s2_bands,
            mosaic_days=mosaic_days,
            step=step,
            cdse_credentials=cdse_credentials,
            ratio=ratio,
            snap_gpt_path=snap_gpt_path,
            log_level=log_level,
            data_source=data_source,
        )
        logger.info("=" * 70)
        logger.info("EFAST processing completed successfully!")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Error during EFAST processing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EFAST algorithm for a selected site and season from selected_sites.geojson"
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default="selected_sites.geojson",
        help="Path to selected_sites.geojson file (default: selected_sites.geojson)",
    )
    parser.add_argument(
        "--sitename",
        type=str,
        required=False,
        help="Name of the site to process (required unless --list-sites is used)",
    )
    parser.add_argument(
        "--season",
        type=str,
        required=False,
        help="Year of the season to process (e.g., '2021') (required unless --list-sites is used)",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List all available sites and seasons, then exit",
    )
    parser.add_argument(
        "--s3-sensor",
        type=str,
        default="SYN",
        help="Sentinel-3 sensor type (default: SYN)",
    )
    parser.add_argument(
        "--s3-bands",
        nargs="+",
        default=["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"],
        help="Sentinel-3 bands (default: SDR_Oa04 SDR_Oa06 SDR_Oa08 SDR_Oa17)",
    )
    parser.add_argument(
        "--s2-bands",
        nargs="+",
        default=["B02", "B03", "B04", "B8A"],
        help="Sentinel-2 bands (default: B02 B03 B04 B8A)",
    )
    parser.add_argument(
        "--mosaic-days",
        type=int,
        default=100,
        help="Mosaic days parameter (default: 100)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Step size in days (default: 2)",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=30,
        help="Resolution ratio (default: 30)",
    )
    parser.add_argument(
        "--snap-gpt-path",
        type=str,
        default="gpt",
        help="Path to SNAP GPT executable (default: gpt)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="cdse",
        choices=["cdse", "stac"],
        help="Data source: 'cdse' for CDSE download or 'stac' for STAC API (default: cdse)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.list_sites and (not args.sitename or not args.season):
        parser.error("--sitename and --season are required unless --list-sites is used")
    
    # Run main wrapper
    main_wrapper(
        geojson_path=args.geojson,
        sitename=args.sitename,
        season_year=args.season,
        s3_sensor=args.s3_sensor,
        s3_bands=args.s3_bands,
        s2_bands=args.s2_bands,
        mosaic_days=args.mosaic_days,
        step=args.step,
        ratio=args.ratio,
        snap_gpt_path=args.snap_gpt_path,
        list_sites=args.list_sites,
        log_level=args.log_level,
        data_source=args.data_source,
    )

