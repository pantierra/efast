"""Helper functions for the interactive notebook interface."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, NamedTuple, List, Any

import ipywidgets as widgets


# Processing constants
DEFAULT_S2_BANDS = ["B02", "B03", "B04", "B8A"]
DEFAULT_S3_BANDS = ["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]
DEFAULT_RATIO = 30
DEFAULT_MOSAIC_DAYS = 100
DEFAULT_STEP = 2
DEFAULT_S3_INSTRUMENT = "OL"
DEFAULT_S3_AGGREGATOR = "mean"
DEFAULT_SMOOTHING_STD = 1

# Spatial extent: ~1km x 1km box around point (0.0045 degrees half-size)
SPATIAL_EXTENT_HALF_SIZE = 0.0045


class NotebookSetup(NamedTuple):
    """Container for notebook setup flags and paths."""
    rasterio_available: bool
    efast_available: bool
    run_efast_available: bool
    test_data_dir: Path


class DataStatus(NamedTuple):
    """Container for data file status counts."""
    s2_raw: int
    s3_raw: int
    s2_processed: int
    s3_reprojected: int
    s2_raw_files: List[Path]
    s3_raw_files: List[Path]


class DataPaths(NamedTuple):
    """Container for data directory paths."""
    s2_raw: Path
    s3_raw: Path
    s2_processed: Path
    s3_reprojected: Path
    s3_binning: Path
    s3_composites: Path
    s3_blurred: Path
    s3_calibrated: Path


def setup_notebook(test_data_path: str = 'test_data') -> NotebookSetup:
    """
    Setup notebook environment: check dependencies and initialize paths.
    
    Returns
    -------
    NotebookSetup
        Named tuple with availability flags and data directory path
    """
    # Check rasterio dependencies
    try:
        import rasterio
        import matplotlib.pyplot as plt
        import numpy as np
        rasterio_available = True
    except ImportError:
        rasterio_available = False
        print("⚠️ Install rasterio for visualization: pip install rasterio matplotlib numpy")
    
    # Check efast package
    try:
        import efast
        efast_available = True
    except ImportError:
        efast_available = False
        print("⚠️ Install efast package: pip install -e .")
    
    # Check run_efast module
    try:
        import run_efast
        # Try to access a function to ensure it's fully importable
        _ = run_efast.get_credentials_from_env
        run_efast_available = True
    except (ImportError, AttributeError) as e:
        run_efast_available = False
        if "creodias" in str(e).lower() or "pystac" in str(e).lower():
            print("⚠️ run_efast dependencies missing. Install: pip install creodias-finder pystac-client shapely tqdm")
    
    return NotebookSetup(
        rasterio_available=rasterio_available,
        efast_available=efast_available,
        run_efast_available=run_efast_available,
        test_data_dir=Path(test_data_path)
    )


def get_data_paths(data_dir: Path, sitename: Optional[str] = None, season_year: Optional[str] = None) -> DataPaths:
    """Get standardized data directory paths, optionally organized by site/season."""
    if sitename and season_year:
        base_dir = data_dir / sitename / season_year
    else:
        base_dir = data_dir
    
    return DataPaths(
        s2_raw=base_dir / 'S2' / 'raw',
        s3_raw=base_dir / 'S3' / 'raw',
        s2_processed=base_dir / 'S2' / 'processed',
        s3_reprojected=base_dir / 'S3' / 'reprojected',
        s3_binning=base_dir / 'S3' / 'binning',
        s3_composites=base_dir / 'S3' / 'composites',
        s3_blurred=base_dir / 'S3' / 'blurred',
        s3_calibrated=base_dir / 'S3' / 'calibrated',
    )


def load_sites(geojson_path: str) -> Dict[str, Dict[str, Any]]:
    """Load sites from GeoJSON file."""
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    sites = {}
    for feature in geojson_data['features']:
        props = feature['properties']
        sitename = props['sitename']
        sites[sitename] = {
            'coordinates': feature['geometry']['coordinates'],
            'description': props.get('description', ''),
            'seasons': props.get('seasons', {})
        }
    return sites


def create_site_selection_widgets(sites: Dict) -> Tuple[widgets.Dropdown, widgets.Dropdown, widgets.Output]:
    """Create site and season selection widgets."""
    site_dropdown = widgets.Dropdown(
        options=sorted(sites.keys()), 
        description='Site:', 
        style={'description_width': '100px'}
    )
    season_dropdown = widgets.Dropdown(
        options=[], 
        description='Season:', 
        style={'description_width': '100px'}
    )
    info_output = widgets.Output()
    
    def update_season(change=None):
        if site_dropdown.value:
            seasons = sorted(sites[site_dropdown.value]['seasons'].keys(), reverse=True)
            season_dropdown.options = seasons
            if seasons:
                season_dropdown.value = seasons[0] if '2024' not in seasons else '2024'
        
        with info_output:
            info_output.clear_output()
            if site_dropdown.value and season_dropdown.value:
                site = sites[site_dropdown.value]
                season_data = site['seasons'][season_dropdown.value]
                print(f"📍 {site_dropdown.value}")
                print(f"📅 {season_data['season_start_date']} to {season_data['season_end_date']}")
                print(f"🛰️ S2: {season_data['sentinel2_scenes']} scenes | S3: {season_data['sentinel3_scenes']} scenes")
    
    site_dropdown.observe(update_season, names='value')
    
    # Initialize
    if 'innsbruck' in site_dropdown.options:
        site_dropdown.value = 'innsbruck'
    elif site_dropdown.options:
        site_dropdown.value = site_dropdown.options[0]
    update_season()
    
    return site_dropdown, season_dropdown, info_output


def get_selected_dates(sites: Dict, sitename: Optional[str], season_year: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Get start and end dates for selected site/season."""
    if not sitename or not season_year or sitename not in sites:
        return None, None
    
    if season_year not in sites[sitename]['seasons']:
        return None, None
    
    season_data = sites[sitename]['seasons'][season_year]
    return season_data.get('season_start_date'), season_data.get('season_end_date')


def check_data_status(data_dir: Path, sitename: Optional[str] = None, season_year: Optional[str] = None) -> DataStatus:
    """Check status of downloaded and processed data files."""
    paths = get_data_paths(data_dir, sitename, season_year)
    
    # Check S2: STAC metadata JSON (for COG processing)
    s2_raw = [paths.s2_raw / "stac_items.json"] if (paths.s2_raw / "stac_items.json").exists() else []
    
    # Check S3: raw files (zip/nc from openEO)
    s3_raw = (list(paths.s3_raw.glob('*.zip')) + 
              list(paths.s3_raw.glob('*.nc'))) if paths.s3_raw.exists() else []
    
    # Check processed files
    s2_processed = list(paths.s2_processed.glob('*REFL.tif')) if paths.s2_processed.exists() else []
    s3_reprojected = list(paths.s3_reprojected.glob('composite*.tif')) if paths.s3_reprojected.exists() else []
    
    return DataStatus(
        s2_raw=len(s2_raw),
        s3_raw=len(s3_raw),
        s2_processed=len(s2_processed),
        s3_reprojected=len(s3_reprojected),
        s2_raw_files=s2_raw,
        s3_raw_files=s3_raw,
    )


def print_status_section(title: str, status: DataStatus, fields: Optional[List[str]] = None):
    """Print a formatted status section."""
    if fields is None:
        fields = ['s2_raw', 's3_raw', 's2_processed', 's3_reprojected']
    
    print("=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    for field in fields:
        count = getattr(status, field)
        status_icon = '✅' if count > 0 else '❌'
        label = field.replace('_', ' ').title()
        print(f"{label}: {status_icon} {count} files")
    print("=" * 60)


def _get_credentials() -> Dict[str, str]:
    """Get CDSE credentials from environment or .env file."""
    try:
        import run_efast
        return run_efast.get_credentials_from_env()
    except Exception:
        raise ValueError("CDSE credentials not found. Set CDSE_USER and CDSE_PASSWORD in environment or .env file")


def _coords_to_wkt(coords: List[float], buffer_degrees: float = SPATIAL_EXTENT_HALF_SIZE) -> str:
    """
    Convert coordinates to WKT POLYGON geometry with a small buffer.
    
    Creates a ~1km x 1km bounding box around the point.
    Used for both S2 (STAC query) and S3 (openEO spatial extent).
    
    Parameters
    ----------
    coords : List[float]
        [longitude, latitude] coordinates
    buffer_degrees : float
        Half-size of the bounding box in degrees (default SPATIAL_EXTENT_HALF_SIZE ~ 1km)
    
    Returns
    -------
    str
        WKT POLYGON string
    """
    lon, lat = coords[0], coords[1]
    west = lon - buffer_degrees
    east = lon + buffer_degrees
    south = lat - buffer_degrees
    north = lat + buffer_degrees
    
    return f"POLYGON (({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"


def download_data_if_needed(sites: Dict[str, Dict[str, Any]], 
                            sitename: Optional[str],
                            season_year: Optional[str],
                            data_dir: Path,
                            run_efast_available: bool) -> bool:
    """Download S2 data. S3 is downloaded via openEO during processing."""
    if not run_efast_available:
        print("❌ run_efast module not available")
        return False
    
    start_date, end_date = get_selected_dates(sites, sitename, season_year)
    if not start_date or not end_date:
        print("❌ Select site and season first")
        return False
    
    status = check_data_status(data_dir, sitename, season_year)
    stac_file = get_data_paths(data_dir, sitename, season_year).s2_raw / "stac_items.json"
    if stac_file.exists():
        import json
        with open(stac_file) as f:
            stac_data = json.load(f)
            print(f"✅ S2 STAC metadata already cached: {len(stac_data)} scenes")
        return False
    
    print(f"📥 Querying S2 COGs from Element84 STAC for {start_date} to {end_date}...")
    
    try:
        from datetime import datetime
        import run_efast
        import json
        
        paths = get_data_paths(data_dir, sitename, season_year)
        coords = sites[sitename]['coordinates']
        aoi_geometry = _coords_to_wkt(coords)
        
        # Show spatial extent
        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(aoi_geometry)
        bbox = geom.bounds
        print(f"📍 Spatial extent: ~1km x 1km box")
        print(f"   Bbox: [{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}]")
        print(f"   Format: Cloud-optimized GeoTIFFs (COGs) from S3")
        
        paths.s2_raw.mkdir(parents=True, exist_ok=True)
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        s2_stac_items = run_efast.query_stac_s2(start_dt, end_dt, aoi_geometry)
        print(f"✅ Found {len(s2_stac_items)} Sentinel-2 COG scenes")
        
        stac_file = paths.s2_raw / "stac_items.json"
        with open(stac_file, 'w') as f:
            json.dump([item.to_dict() for item in s2_stac_items], f, indent=2)
        print(f"💾 Saved STAC metadata: {stac_file}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_download_button(sites: Dict[str, Dict[str, Any]], 
                          site_dropdown: widgets.Dropdown, 
                          season_dropdown: widgets.Dropdown, 
                          data_dir: Path, 
                          run_efast_available: bool) -> Tuple[widgets.Button, widgets.Output]:
    """Create download button with handler."""
    download_output = widgets.Output()
    download_button = widgets.Button(
        description='📥 Download Data',
        button_style='primary',
        disabled=not run_efast_available
    )
    
    def run_download(b):
        with download_output:
            download_output.clear_output()
            sitename = site_dropdown.value
            season_year = season_dropdown.value
            status = check_data_status(data_dir, sitename, season_year)
            cached_s2 = status.s2_raw
            cached_s3 = status.s3_raw
            
            if cached_s2 > 0 or cached_s3 > 0:
                print(f"📦 Cached: S2={cached_s2} files, S3={cached_s3} files")
                print("💡 Re-downloading (will overwrite cache)...")
            
            download_data_if_needed(
                sites, sitename, season_year,
                data_dir, run_efast_available
            )
    
    download_button.on_click(run_download)
    return download_button, download_output


def _run_s2_processing(paths: DataPaths) -> str:
    """Run S2 processing pipeline using STAC COGs. Returns footprint WKT."""
    import efast.s2_processing as s2
    from pystac import Item
    
    print("\n🛰️ Processing S2 from STAC COGs...")
    paths.s2_processed.mkdir(parents=True, exist_ok=True)
    
    # Load STAC items if available
    stac_file = paths.s2_raw / "stac_items.json"
    stac_items = None
    if stac_file.exists():
        import json
        with open(stac_file) as f:
            stac_dicts = json.load(f)
            stac_items = [Item.from_dict(d) for d in stac_dicts]
        print(f"📂 Loading {len(stac_items)} COG scenes from S3 (no download needed)")
    
    s2.extract_mask_s2_bands(paths.s2_raw, paths.s2_processed, 
                             bands=DEFAULT_S2_BANDS, stac_items=stac_items)
    s2.distance_to_clouds(paths.s2_processed, ratio=DEFAULT_RATIO)
    return s2.get_wkt_footprint(paths.s2_processed)


def _run_s3_processing(paths: DataPaths, footprint: str,
                       geojson_path: str, sitename: str, season_year: str) -> None:
    """Run S3 processing pipeline using openEO (spatially subsetted ~1km x 1km)."""
    import efast.s3_processing as s3
    
    print("\n🌊 Processing S3 via openEO...")
    for d in [paths.s3_binning, paths.s3_composites, paths.s3_blurred, 
              paths.s3_calibrated, paths.s3_reprojected]:
        d.mkdir(parents=True, exist_ok=True)
    
    s3.binning_s3(paths.s3_raw, paths.s3_binning, footprint=footprint,
                  s3_bands=DEFAULT_S3_BANDS, instrument=DEFAULT_S3_INSTRUMENT,
                  aggregator=DEFAULT_S3_AGGREGATOR,
                  use_openeo=True,
                  geojson_path=geojson_path,
                  site_name=sitename,
                  season_year=season_year)
    s3.produce_median_composite(paths.s3_binning, paths.s3_composites, 
                                mosaic_days=DEFAULT_MOSAIC_DAYS, step=DEFAULT_STEP)
    s3.smoothing(paths.s3_composites, paths.s3_blurred, std=DEFAULT_SMOOTHING_STD, 
                 preserve_nan=False)
    s3.reformat_s3(paths.s3_blurred, paths.s3_calibrated)
    s3.reproject_and_crop_s3(paths.s3_calibrated, paths.s2_processed, paths.s3_reprojected)


def create_processing_button(sites: Dict[str, Dict[str, Any]],
                            site_dropdown: widgets.Dropdown,
                            season_dropdown: widgets.Dropdown,
                            data_dir: Path, 
                            efast_available: bool) -> Tuple[widgets.Button, widgets.Output]:
    """Create processing button with handler."""
    process_output = widgets.Output()
    
    def run_processing(b):
        with process_output:
            process_output.clear_output()
            
            sitename = site_dropdown.value
            season_year = season_dropdown.value
            status = check_data_status(data_dir, sitename, season_year)
            
            if not status.s2_raw_files:
                print("❌ Download S2 data first")
                return
            
            print("⚙️ Processing data...")
            paths = get_data_paths(data_dir, sitename, season_year)
            footprint = _run_s2_processing(paths)
            _run_s3_processing(paths, footprint, "selected_sites.geojson", sitename, season_year)
            print("\n✅ Processing complete!")
    
    status = check_data_status(data_dir)
    has_raw_data = bool(status.s2_raw_files and status.s3_raw_files)
    is_disabled = not (efast_available and has_raw_data)
    
    # If processed files exist but no raw files, provide helpful message
    if status.s2_processed > 0 or status.s3_reprojected > 0:
        if not has_raw_data:
            # Note: This info will be shown when button is clicked, but we can't show it here
            # The button will be disabled, which is correct - need raw files to process
            pass
    
    process_button = widgets.Button(
        description='⚙️ Process Data',
        button_style='info',
        disabled=is_disabled,
        tooltip='Requires raw S2 and S3 data files' if is_disabled else 'Click to process data'
    )
    
    process_button.on_click(run_processing)
    return process_button, process_output

