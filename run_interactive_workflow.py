#!/usr/bin/env python3
"""
Non-interactive script version of the efast_interactive notebook.
Can be used for automated testing and CI/CD.

Usage:
    python run_interactive_workflow.py --sitename innsbruck --season 2024
    python run_interactive_workflow.py --sitename innsbruck --season 2024 --download-only
    python run_interactive_workflow.py --sitename innsbruck --season 2024 --process-only
"""

import argparse
import os
from pathlib import Path

# Load .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = value

from efast.interactive_helpers import (
    setup_notebook, load_sites, check_data_status, print_status_section,
    download_data_if_needed, get_data_paths
)
import efast.s2_processing as s2
import efast.s3_processing as s3
from efast.interactive_helpers import (
    DEFAULT_S2_BANDS, DEFAULT_S3_BANDS, DEFAULT_RATIO, DEFAULT_MOSAIC_DAYS,
    DEFAULT_STEP, DEFAULT_S3_INSTRUMENT, DEFAULT_S3_AGGREGATOR, DEFAULT_SMOOTHING_STD
)


def run_processing(data_dir: Path, sitename: str, season_year: str):
    """Run S2 and S3 processing pipeline."""
    from efast.interactive_helpers import _run_s2_processing, _run_s3_processing
    
    paths = get_data_paths(data_dir, sitename, season_year)
    status = check_data_status(data_dir, sitename, season_year)
    
    if not status.s2_raw_files:
        print("❌ Download S2 data first")
        return False
    
    print("⚙️ Processing data...")
    
    try:
        footprint = _run_s2_processing(paths)
        _run_s3_processing(paths, footprint, "selected_sites.geojson", sitename, season_year)
        print("\n✅ Processing complete!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run eFAST workflow non-interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--sitename', required=True,
        help='Site name (e.g., innsbruck)'
    )
    parser.add_argument(
        '--season', required=True,
        help='Season year (e.g., 2024)'
    )
    parser.add_argument(
        '--data-dir', default='test_data',
        help='Data directory (default: test_data)'
    )
    parser.add_argument(
        '--geojson', default='selected_sites.geojson',
        help='Path to GeoJSON file (default: selected_sites.geojson)'
    )
    parser.add_argument(
        '--download-only', action='store_true',
        help='Only download data, skip processing'
    )
    parser.add_argument(
        '--process-only', action='store_true',
        help='Only process data, skip download'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate inputs and check status without downloading/processing'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup = setup_notebook(args.data_dir)
    if not setup.run_efast_available:
        print("❌ run_efast module not available. Install dependencies first.")
        return 1
    
    # Load sites
    sites = load_sites(args.geojson)
    if args.sitename not in sites:
        print(f"❌ Site '{args.sitename}' not found in {args.geojson}")
        print(f"Available sites: {', '.join(sorted(sites.keys()))}")
        return 1
    
    if args.season not in sites[args.sitename]['seasons']:
        print(f"❌ Season '{args.season}' not found for site '{args.sitename}'")
        print(f"Available seasons: {', '.join(sorted(sites[args.sitename]['seasons'].keys()))}")
        return 1
    
    data_dir = Path(args.data_dir)
    print(f"📍 Site: {args.sitename}")
    print(f"📅 Season: {args.season}")
    print(f"💾 Cache location: {data_dir / args.sitename / args.season}")
    print("=" * 60)
    
    # Download step
    if not args.process_only:
        print("\n## Step 1: Get Data")
        print("=" * 60)
        status = check_data_status(data_dir, args.sitename, args.season)
        print_status_section("DATA STATUS", status, fields=['s2_raw', 's3_raw'])
        
        if args.dry_run:
            if status.s2_raw == 0 and status.s3_raw == 0:
                print("\n💡 Dry run: Would download data (no cached data found)")
            else:
                print(f"\n💡 Dry run: Data already cached (S2={status.s2_raw}, S3={status.s3_raw})")
        elif status.s2_raw == 0 and status.s3_raw == 0:
            print("\n📥 No cached data found. Downloading...")
            success = download_data_if_needed(
                sites, args.sitename, args.season,
                data_dir, setup.run_efast_available
            )
            if not success:
                print("❌ Download failed")
                return 1
            
            # Refresh status
            status = check_data_status(data_dir, args.sitename, args.season)
            print_status_section("DATA STATUS (after download)", status, fields=['s2_raw', 's3_raw'])
        else:
            print(f"\n✅ S2 STAC metadata already cached: {status.s2_raw} file")
    
    # Processing step
    if not args.download_only:
        print("\n## Step 2: Processing")
        print("=" * 60)
        status = check_data_status(data_dir, args.sitename, args.season)
        print_status_section("PROCESSING STATUS", status, fields=['s2_processed', 's3_reprojected'])
        
        if args.dry_run:
            if status.s2_processed == 0 or status.s3_reprojected == 0:
                if status.s2_raw > 0:
                    print("\n💡 Dry run: Would process data (S2 STAC metadata available)")
                else:
                    print("\n💡 Dry run: Cannot process (no S2 STAC metadata)")
            else:
                print(f"\n💡 Dry run: Processing already done (S2={status.s2_processed}, S3={status.s3_reprojected})")
        elif status.s2_processed == 0 or status.s3_reprojected == 0:
            success = run_processing(data_dir, args.sitename, args.season)
            if not success:
                print("❌ Processing failed")
                return 1
            
            # Refresh status
            status = check_data_status(data_dir, args.sitename, args.season)
            print_status_section("PROCESSING STATUS (after processing)", status, 
                                fields=['s2_processed', 's3_reprojected'])
        else:
            print(f"\n✅ Processing already done: S2={status.s2_processed} files, "
                  f"S3={status.s3_reprojected} files")
    
    print("\n" + "=" * 60)
    print("✅ Workflow complete!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())

