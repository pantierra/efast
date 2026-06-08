"""Query the Copernicus Data Space Ecosystem catalogue via OData."""

from datetime import date, datetime, time, timedelta
from urllib.parse import quote

import requests

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def _format_odata_datetime(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    else:
        dt = datetime.strptime(str(value), "%Y-%m-%d")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _end_of_day(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    else:
        dt = datetime.strptime(str(value), "%Y-%m-%d")
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        dt = dt + timedelta(hours=23, minutes=59, seconds=59)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.999Z")


def _string_attribute(name, value):
    return (
        "Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq '{name}' and att/OData.CSC.StringAttribute/Value eq '{value}'"
        ")"
    )


def query_products(
    collection,
    start_date,
    end_date,
    geometry,
    *,
    product_type=None,
    timeliness=None,
    online=True,
    page_size=1000,
):
    """Query CDSE OData and return products keyed by UUID.

    Parameters
    ----------
    collection : str
        OData collection name, e.g. ``SENTINEL-2`` or ``SENTINEL-3``.
    start_date, end_date : datetime.date, datetime.datetime, or str
        Acquisition date range (inclusive).
    geometry : str
        Area of interest as WKT, e.g. ``POINT (-71.28 44.06)``.
    product_type : str, optional
        Product type attribute value, e.g. ``S2MSI2A`` or ``SY_2_SYN___``.
    timeliness : str, optional
        Timeliness attribute value, e.g. ``NT``.
    online : bool, optional
        If True, restrict to online products.
    page_size : int, optional
        Number of records per OData page.

    Returns
    -------
    dict[str, dict]
        Products keyed by UUID. Each value includes an ``id`` field for
        compatibility with legacy creodias-finder callers.
    """
    filters = [
        f"Collection/Name eq '{collection}'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{geometry}')",
        f"ContentDate/Start ge {_format_odata_datetime(start_date)}",
        f"ContentDate/Start le {_end_of_day(end_date)}",
    ]
    if product_type is not None:
        filters.append(_string_attribute("productType", product_type))
    if timeliness is not None:
        filters.append(_string_attribute("timeliness", timeliness))
    if online:
        filters.append("Online eq true")

    query_url = (
        f"{CATALOGUE_URL}?$filter={quote(' and '.join(filters), safe='')}"
        f"&$top={page_size}"
    )

    products = {}
    while query_url:
        response = requests.get(query_url, timeout=120)
        response.raise_for_status()
        payload = response.json()

        for product in payload["value"]:
            product_id = product["Id"]
            products[product_id] = {"id": product_id, **product}

        query_url = payload.get("@odata.nextLink")

    return products
