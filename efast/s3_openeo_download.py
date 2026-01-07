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

import json
import logging
import os
from datetime import datetime

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import openeo
import rasterio
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


def map_bands_to_openeo(bands: List[str]) -> List[str]:
    """
    Map EFAST band names (SDR_OaXX) to OpenEO band names (BXX).
    
    Parameters
    ----------
    bands : List[str]
        List of band names in EFAST format (e.g., ['SDR_Oa04', 'SDR_Oa06'])
    
    Returns
    -------
    List[str]
        List of band names in OpenEO format (e.g., ['B04', 'B06'])
    """
    band_mapping = {
        "SDR_Oa01": "B01", "SDR_Oa02": "B02", "SDR_Oa03": "B03",
        "SDR_Oa04": "B04", "SDR_Oa05": "B05", "SDR_Oa06": "B06",
        "SDR_Oa07": "B07", "SDR_Oa08": "B08", "SDR_Oa09": "B09",
        "SDR_Oa10": "B10", "SDR_Oa11": "B11", "SDR_Oa12": "B12",
        "SDR_Oa13": "B13", "SDR_Oa14": "B14", "SDR_Oa15": "B15",
        "SDR_Oa16": "B16", "SDR_Oa17": "B17", "SDR_Oa18": "B18",
        "SDR_Oa19": "B19", "SDR_Oa20": "B20", "SDR_Oa21": "B21",
    }
    
    mapped_bands = []
    for band in bands:
        if band in band_mapping:
            mapped_bands.append(band_mapping[band])
        elif band.startswith("B") and len(band) <= 3:
            # Already in OpenEO format
            mapped_bands.append(band)
        else:
            logger.warning(f"Unknown band name: {band}, using as-is")
            mapped_bands.append(band)
    
    return mapped_bands


class S3OpenEODownloader:
    """Download Sentinel-3 OLCI data from CDSE using openEO API."""

    def __init__(self):
        self.endpoint = "openeo.dataspace.copernicus.eu"
        # Use L1B collection which has the raw OLCI bands (B01-B21)
        # L2 collections have processed bands that don't match SDR_OaXX naming
        self.collection_id = "SENTINEL3_OLCI_L1B"
        self.connection = None

    def authenticate(self) -> bool:
        """Authenticate with CDSE openEO service using OIDC.
        
        This will attempt to use stored credentials first, or prompt for browser-based
        authentication if needed. The browser will open automatically for authentication.
        
        Returns
        -------
        bool
            True if authentication successful, False otherwise
        """
        try:
            logger.info(f"Connecting to openEO endpoint: {self.endpoint}")
            self.connection = openeo.connect(self.endpoint)
            
            logger.info("Authenticating with CDSE openEO using OIDC...")
            logger.info("If a browser window opens, please complete the authentication there.")
            logger.info("If no browser opens, check the console for a URL to visit manually.")
            
            self.connection.authenticate_oidc()
            
            # Verify authentication by checking if we can access the connection
            if self.connection:
                logger.info("Successfully authenticated with CDSE openEO")
                return True
            else:
                logger.error("Authentication completed but connection is not available")
                return False
                
        except KeyboardInterrupt:
            logger.warning("Authentication cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.error("Please ensure you have:")
            logger.error("  1. A valid CDSE account (register at https://identity.dataspace.copernicus.eu/)")
            logger.error("  2. Network access to openEO endpoint")
            logger.error("  3. A browser available for OIDC authentication")
            return False

    def download_from_geojson_site(
        self,
        geojson_path: str,
        site_name: str,
        season_year: str,
        output_dir: Path,
        bands: List[str] = ["SDR_Oa04", "SDR_Oa06", "SDR_Oa08", "SDR_Oa17"],
    ) -> Optional[str]:
        """Download S3 data for a site defined in GeoJSON."""
        if not self.connection:
            logger.error("Not authenticated")
            return None

        try:
            # Load site data
            sites_gdf = gpd.read_file(geojson_path)
            site_data = sites_gdf[sites_gdf["sitename"] == site_name]
            if site_data.empty:
                logger.error(f"Site '{site_name}' not found")
                return None

            site_row = site_data.iloc[0]
            center_lon = site_row.geometry.x
            center_lat = site_row.geometry.y

            # Parse seasons - geopandas may read it as a JSON string
            seasons_raw = site_row["seasons"]
            if isinstance(seasons_raw, str):
                seasons = json.loads(seasons_raw)
            else:
                seasons = seasons_raw
            
            if season_year not in seasons:
                logger.error(f"Season {season_year} not found")
                return None

            season_data = seasons[season_year]
            start_date = season_data["season_start_date"]
            end_date = season_data["season_end_date"]

            # Create spatial extent (1km x 1km)
            half_size = 0.0045  # ~1km in degrees
            spatial_extent = {
                "west": center_lon - half_size,
                "east": center_lon + half_size,
                "south": center_lat - half_size,
                "north": center_lat + half_size,
            }

            # Map band names to OpenEO format
            openeo_bands = map_bands_to_openeo(bands)
            logger.info(f"Mapped bands {bands} to OpenEO bands {openeo_bands}")

            # Load and process data
            datacube = self.connection.load_collection(
                collection_id=self.collection_id,
                spatial_extent=spatial_extent,
                temporal_extent=[start_date, end_date],
                bands=openeo_bands,
            )

            # Output file
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"S3_OLCI_{site_name}_{season_year}.nc"

            # Download
            datacube.download(str(output_file), format="NetCDF")
            logger.info(f"Downloaded: {output_file}")
            
            # Convert NetCDF to individual TIFF files
            tif_files = self.convert_netcdf_to_tiffs(output_file, output_dir, bands)
            logger.info(f"Converted NetCDF to {len(tif_files)} TIFF files")
            
            return str(output_file)

        except Exception as e:
            logger.error(f"Download failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def convert_netcdf_to_tiffs(
        self,
        nc_file: Path,
        output_dir: Path,
        original_bands: List[str],
    ) -> List[Path]:
        """
        Convert NetCDF file from OpenEO to individual TIFF files per time step.
        
        Parameters
        ----------
        nc_file : Path
            Path to the NetCDF file
        output_dir : Path
            Directory to save TIFF files
        original_bands : List[str]
            Original band names (for reference)
        
        Returns
        -------
        List[Path]
            List of created TIFF file paths
        """
        try:
            import netCDF4
            
            logger.info(f"Converting NetCDF file: {nc_file}")
            nc = netCDF4.Dataset(str(nc_file), 'r')
            
            # Get dimensions
            time_var = nc.variables.get('t')
            if time_var is None:
                logger.error("No time dimension found in NetCDF")
                nc.close()
                return []
            
            times = netCDF4.num2date(time_var[:], time_var.units)
            x_var = nc.variables.get('x')
            y_var = nc.variables.get('y')
            
            if x_var is None or y_var is None:
                logger.error("Missing x or y dimensions in NetCDF")
                nc.close()
                return []
            
            # Get band variables (should be B04, B06, B08, B17)
            band_vars = [var for var in nc.variables.keys() 
                        if var.startswith('B') and var[1:].isdigit()]
            band_vars.sort()  # Sort to maintain order
            
            logger.info(f"Found {len(band_vars)} bands: {band_vars}")
            logger.info(f"Found {len(times)} time steps")
            
            # Get spatial extent from coordinates
            x_coords = x_var[:]
            y_coords = y_var[:]
            
            # Create transform
            west = float(x_coords.min())
            east = float(x_coords.max())
            south = float(y_coords.min())
            north = float(y_coords.max())
            
            width = len(x_coords)
            height = len(y_coords)
            
            transform = from_bounds(west, south, east, north, width, height)
            
            tif_files = []
            
            # Process each time step
            for t_idx, time_val in enumerate(times):
                # Format date as YYYYMMDDTHHMMSS
                if isinstance(time_val, datetime):
                    date_str = time_val.strftime("%Y%m%dT%H%M%S")
                else:
                    # Handle numpy datetime64
                    dt = netCDF4.num2date(time_var[t_idx], time_var.units)
                    date_str = dt.strftime("%Y%m%dT%H%M%S")
                
                # Create output filename matching expected pattern
                # Pattern: S3*__YYYYMMDDTHHMMSS.tif
                output_filename = f"S3_OLCI__{date_str}.tif"
                output_path = output_dir / output_filename
                
                # Read all bands for this time step
                band_data = []
                for band_var in band_vars:
                    var_data = nc.variables[band_var]
                    # NetCDF dimensions are typically (t, y, x) or (t, bands, y, x)
                    if len(var_data.shape) == 3:
                        # (t, y, x)
                        band_data.append(var_data[t_idx, :, :])
                    elif len(var_data.shape) == 4:
                        # (t, bands, y, x) - unlikely but handle it
                        band_data.append(var_data[t_idx, 0, :, :])
                    else:
                        logger.warning(f"Unexpected shape for {band_var}: {var_data.shape}")
                        continue
                
                if not band_data:
                    logger.warning(f"No band data for time step {t_idx}")
                    continue
                
                # Stack bands (bands, height, width)
                stacked = np.stack(band_data, axis=0)
                
                # Write TIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=len(band_data),
                    dtype=stacked.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    compress='lzw',
                ) as dst:
                    dst.write(stacked)
                
                tif_files.append(output_path)
            
            nc.close()
            logger.info(f"Created {len(tif_files)} TIFF files")
            return tif_files
            
        except Exception as e:
            logger.error(f"Error converting NetCDF to TIFFs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
