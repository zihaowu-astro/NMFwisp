#!/usr/bin/env python
"""Download example SW and LW NIRCam data for NMFwisp.

The SW exposure is used as the target image for wisp subtraction. The LW
exposures are used to build the source mask segmentation map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astroquery.mast import MastMissions


SW_DATASETS = ["jw01286008001_31201_00001"]
SW_PRODUCTS = ["jw01286008001_31201_00001_jw01286008001_31201_00001_nrcb4_cal.fits"]

LW_DATASETS = ["jw01286008001_31201_00001", "jw01286008001_31201_00002", "jw01286008001_31201_00003"]
LW_PRODUCTS = [
    "jw01286008001_31201_00001_jw01286008001_31201_00001_nrcalong_cal.fits",
    "jw01286008001_31201_00002_jw01286008001_31201_00002_nrcalong_cal.fits",
    "jw01286008001_31201_00003_jw01286008001_31201_00003_nrcalong_cal.fits",
    "jw01286008001_31201_00001_jw01286008001_31201_00001_nrcblong_cal.fits",
    "jw01286008001_31201_00002_jw01286008001_31201_00002_nrcblong_cal.fits",
    "jw01286008001_31201_00003_jw01286008001_31201_00003_nrcblong_cal.fits",
]


def download_products(missions, datasets, product_keys, download_dir):
    download_dir.mkdir(parents=True, exist_ok=True)
    products = missions.get_product_list(datasets)
    filtered = missions.filter_products(products, product_key=product_keys)
    result = missions.download_products(filtered, download_dir=str(download_dir), flat=True)
    return len(result)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Download example SW and LW NIRCam data for NMFwisp.",)
    parser.add_argument(
        "--data_dir",
        type=Path, default=Path("./data"), help="Output directory for the SW target exposure.",)
    parser.add_argument(
        "--lw_dir",
        type=Path, default=None, help="Output directory for LW images. Defaults to DATA_DIR/LW_data.",)
    return parser


def main():
    args = get_parser().parse_args()
    data_dir = args.data_dir
    lw_dir = args.lw_dir or data_dir / "LW_data"

    missions = MastMissions(mission="JWST")
    n_sw = download_products(missions, SW_DATASETS, SW_PRODUCTS, data_dir)
    n_lw = download_products(missions, LW_DATASETS, LW_PRODUCTS, lw_dir)

    print(f"Downloaded {n_sw} SW file(s) to {data_dir}")
    print(f"Downloaded {n_lw} LW file(s) to {lw_dir}")


if __name__ == "__main__":
    main()
