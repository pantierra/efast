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

import logging
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pyproj
import rasterio
import scipy as sp

from shapely.geometry import box
from shapely.ops import transform
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Mapping of Sentinel-2 bands names to bands ids
BANDS_IDS = {
    "B02": "1",
    "B03": "2",
    "B04": "3",
    "B05": "4",
    "B06": "5",
    "B07": "6",
    "B08": "7",
    "B8A": "8",
    "B11": "11",
    "B12": "12",
}

# Mapping of Sentinel-2 band names to STAC asset keys (Element84 naming)
STAC_BAND_MAPPING = {
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
    "SCL": "scl",
}


def extract_mask_s2_bands(
    input_dir,
    output_dir,
    bands=["B02", "B03", "B04", "B8A"],
    resolution=20,
    stac_items=None,
):
    """
    Extract specified Sentinel-2 bands from .SAFE file or STAC items, mask clouds and shadows using the SLC mask
    and save to multi-band GeoTIFF file.

    Parameters
    ----------
    input_dir : pathlib.Path
        The directory where the Sentinel-2 .SAFE images are stored (used if stac_items is None).
    output_dir: pathlib.Path
        The directory where the Sentinel-2 GeoTIFF images are to be stored.
    bands: list [str], optional
        List of bands names to be extracted from the .SAFE file.
        Defaults to ["B02", "B03", "B04", "B08"]
    resolution: int, optional
        Spatial resolution of the bands to be extracted.
        Defaults to 20.
    stac_items: list, optional
        List of STAC items to process. If provided, processes COGs from STAC instead of .SAFE files.

    Returns
    -------
    None
    """
    # Process STAC items if provided
    if stac_items is not None:
        # Configure rasterio environment for unsigned access to AWS S3 COGs
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
            for item in tqdm(stac_items, desc="Processing STAC items"):
                # Get COG asset URLs from STAC item
                # Element84 STAC uses asset keys like "B02", "B03", etc. and "SCL" for cloud mask
                try:
                    # Find first available band to get profile using STAC band mapping
                    profile = None
                    first_band = bands[0]
                    # Map band name to STAC asset key
                    stac_band_key = STAC_BAND_MAPPING.get(
                        first_band, first_band.lower()
                    )

                    # Try STAC mapping first, then fallback to original name variants
                    for key_variant in [
                        stac_band_key,
                        first_band,
                        first_band.upper(),
                        first_band.lower(),
                    ]:
                        if key_variant in item.assets:
                            first_band_url = item.assets[key_variant].href
                            with rasterio.open(first_band_url) as src:
                                profile = src.profile.copy()
                            break

                    if profile is None:
                        logger.warning(
                            f"Could not find band {first_band} in STAC item {item.id}"
                        )
                        logger.warning(
                            f"Available assets: {list(item.assets.keys())[:15]}"
                        )
                        continue

                    # Read SCL cloud mask - use STAC mapping
                    scl_key = STAC_BAND_MAPPING.get("SCL", "scl")
                    scl_url = None
                    for scl_variant in [
                        scl_key,
                        "SCL",
                        "scl",
                        "scene-classification",
                    ]:
                        if scl_variant in item.assets:
                            scl_url = item.assets[scl_variant].href
                            break

                    if scl_url is None:
                        logger.warning(
                            f"SCL band not found in STAC item {item.id}"
                        )
                        logger.warning(
                            f"Available assets: {list(item.assets.keys())[:15]}"
                        )
                        continue

                    # Read SCL mask and ensure it matches the profile dimensions
                    with rasterio.open(scl_url) as mask_src:
                        mask_data = mask_src.read(1)
                        mask_profile = mask_src.profile

                    # If mask has different dimensions, resample it to match the band profile
                    if mask_data.shape != (profile["height"], profile["width"]):
                        from rasterio.warp import Resampling, reproject

                        mask_resampled = np.zeros(
                            (profile["height"], profile["width"]),
                            dtype=mask_data.dtype,
                        )
                        reproject(
                            source=mask_data,
                            destination=mask_resampled,
                            src_transform=mask_profile["transform"],
                            src_crs=mask_profile["crs"],
                            dst_transform=profile["transform"],
                            dst_crs=profile["crs"],
                            resampling=Resampling.nearest,
                        )
                        mask = mask_resampled
                    else:
                        mask = mask_data

                    mask = (mask == 0) | (mask == 3) | (mask > 7)

                    # Combine bands and mask
                    s2_image = np.zeros(
                        (len(bands), profile["height"], profile["width"]),
                        "float32",
                    )

                    for i, band in enumerate(bands):
                        # Map band name to STAC asset key
                        stac_band_key = STAC_BAND_MAPPING.get(
                            band, band.lower()
                        )
                        band_url = None
                        # Try STAC mapping first, then fallback to original name variants
                        for key_variant in [
                            stac_band_key,
                            band,
                            band.upper(),
                            band.lower(),
                        ]:
                            if key_variant in item.assets:
                                band_url = item.assets[key_variant].href
                                break

                        if band_url is None:
                            logger.warning(
                                f"Band {band} not found in STAC item {item.id}, skipping"
                            )
                            continue

                        with rasterio.open(band_url) as src:
                            # STAC COGs from Element84 are typically in reflectance (0-1) or DN*10000
                            # Check if values need scaling
                            data = src.read(1).astype("float32")

                            # Ensure data matches profile dimensions (resample if needed)
                            if data.shape != (
                                profile["height"],
                                profile["width"],
                            ):
                                from rasterio.warp import Resampling, reproject

                                data_resampled = np.zeros(
                                    (profile["height"], profile["width"]),
                                    dtype=data.dtype,
                                )
                                reproject(
                                    source=data,
                                    destination=data_resampled,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=profile["transform"],
                                    dst_crs=profile["crs"],
                                    resampling=Resampling.bilinear,
                                )
                                data = data_resampled

                            if data.max() > 1.0:
                                # Likely DN values, scale to reflectance
                                data = data / 10000.0
                            # Ensure values are in [0, 1] range
                            data = np.clip(data, 0, 1)
                            data[mask] = 0
                            s2_image[i] = data

                    # Save file
                    profile.update(
                        {
                            "driver": "GTiff",
                            "count": len(bands),
                            "dtype": "float32",
                            "nodata": 0,
                        }
                    )
                    # Use STAC item ID for filename
                    out_path = output_dir / f"{item.id}_REFL.tif"
                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(s2_image)

                except Exception as e:
                    logger.error(f"Error processing STAC item {item.id}: {e}")
                    import traceback

                    logger.debug(traceback.format_exc())
                    continue

        return

    # Original .SAFE file processing
    for p in input_dir.glob("*.SAFE"):
        band_paths = [
            list(p.glob(f"GRANULE/*/IMG_DATA/R{resolution}m/*{band}*.jp2"))[0]
            for band in bands
        ]

        # Find S2 BOA offsets
        tree = ET.parse(p / "MTD_MSIL2A.xml")
        root = tree.getroot()
        offset_list = root.findall(".//BOA_ADD_OFFSET")
        offsets = {el.attrib["band_id"]: el.text for el in offset_list}

        # Extract rasterio profile
        with rasterio.open(band_paths[0]) as src:
            profile = src.profile.copy()

        # Read SLC cloud mask
        mask_path = list(p.glob(f"GRANULE/*/IMG_DATA/R{resolution}m/*SCL*"))[0]
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)
        mask = (mask == 0) | (mask == 3) | (mask > 7)

        # Combine bands and mask
        s2_image = np.zeros(
            (len(bands), profile["height"], profile["width"]), "float32"
        )
        for i, band_path in enumerate(band_paths):
            band = bands[i]
            band_id = BANDS_IDS.get(band)
            offset = int(offsets.get(band_id, 0))
            with rasterio.open(band_path) as src:
                raw_data = src.read(1).astype("int16")
                data = (raw_data + offset) / 10000
                data[data < 0] = 0
                data[mask] = 0
                s2_image[i] = data

        # Save file
        profile.update(
            {
                "driver": "GTiff",
                "count": len(bands),
                "dtype": "float32",
                "nodata": 0,
            }
        )
        out_path = output_dir / f"{str(p.name).rstrip('.SAFE')}_REFL.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(s2_image)


def distance_to_clouds(dir_s2, ratio=30, tolerance_percentage=0.05):
    """
    Calculate distance to nearest cloud (or other no-data part of the image) and save it at
    roughly OLCI spatial resolution (300 m)

    Parameters
    ----------
    dir_s2 : pathlib.Path
        The directory where the Sentinel-2 images are stored. Clouds and shadows should the masked
        using 0.
    ratio: int, optional
        The (rough) ratio between resolution of Sentinel-2 and Sentinel-3 images. Defaults to 30.
    tolerance_percentage: float, optional
        Fraction of low-resolution (Sentinel-3) pixel which can be covered by Sentinel-2 resolution
        cloudy pixels before the low-resolution pixel is considered to be cloudy. Defaults to 0.05.

    Returns
    -------
    None

    References
    ----------
    ..  [Senty2024] Senty, P., Guzinski, R., Grogan, K., Buitenwerf, R., Ardö, J., Eklundh, L.,
        Koukos, A., Tagesson, T., and Munk, M. (2024). Fast Fusion of Sentinel-2 and Sentinel-3
        Time Series over Rangelands. Remote Sensing 16, 1833. https://doi.org/10.3390/rs16111833
    ..  [Griffiths2013] Griffiths, P.; van der Linden, S.; Kuemmerle, T.; Hostert, P. A Pixel-Based
        Landsat Compositing Algorithm for Large Area Land Cover Mapping. IEEE J. Sel. Top. Appl.
        Earth Obs. Remote Sens. 2013, 6, 2088–2101. https://doi.org/10.1109/JSTARS.2012.2228167.
    """

    sen2_paths = dir_s2.glob("*REFL.tif")

    for sen2_path in tqdm(sen2_paths):
        # Read s2 image
        with rasterio.open(sen2_path) as src:
            s2_hr = src.read(1)
            s2_profile = src.profile

        # Check if a Sentinel-3 pixel is complete
        s2_block = (
            (s2_hr == 0)
            .reshape(
                s2_hr.shape[0] // ratio, ratio, s2_hr.shape[1] // ratio, ratio
            )
            .mean(3)
            .mean(1)
        )

        # Distance to cloud score
        mask = s2_block < tolerance_percentage
        distance_to_cloud = sp.ndimage.distance_transform_edt(mask)
        distance_to_cloud = np.clip(distance_to_cloud, 0, 255)

        # Update transform
        s2_resolution = (s2_profile["transform"] * (1, 0))[0] - (
            s2_profile["transform"] * (0, 0)
        )[0]
        longitude_origin, latitude_origin = s2_profile["transform"] * (0, 0)
        lr_transform = rasterio.Affine(
            ratio * s2_resolution,
            0,
            longitude_origin,
            0,
            -ratio * s2_resolution,
            latitude_origin,
        )

        # Update profile to sentinel-3 geometry
        s2_profile.update(
            {
                "width": mask.shape[1],
                "height": mask.shape[0],
                "transform": lr_transform,
            }
        )

        # Update profile with a new dtype
        s2_profile.update({"count": 1})

        # Save output
        out_path = re.sub(r"_[A-Z]*\.tif", "_DIST_CLOUD.tif", str(sen2_path))
        with rasterio.open(out_path, "w", **s2_profile) as dst:
            dst.write(distance_to_cloud[np.newaxis])


def get_wkt_footprint(dir_s2, crs="EPSG:4326"):
    """
    Get the footprint (bounds) of the first image in the directory in WKT format

    Parameters
    ----------
    dir_s2 : pathlib.Path
        The directory where the Sentinel-2 images are stored.
    crs: str, optional
        The projection of the returned footrpint. Defaults to EPSG:4326

    Returns
    -------
    footprint: str
        The footprint in WKT format
    """

    image_path = list(dir_s2.glob("*REFL.tif"))[0]

    # Get images's bounds and CRS
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        image_crs = src.crs

    # Ensure footprint is in desired CRS
    polygon = box(*bounds)
    if image_crs != crs:
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(image_crs), pyproj.Proj(crs), always_xy=True
        )
        polygon = transform(transformer.transform, polygon)

    # Step 4: Convert to WKT
    footprint = polygon.wkt

    return footprint
