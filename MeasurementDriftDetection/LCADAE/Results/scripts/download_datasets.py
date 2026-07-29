#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAS_URL = "https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip"
SKAB_URL = "https://codeload.github.com/waico/SKAB/zip/refs/heads/master"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "cross-domain-drift-research/1.0"}
    with requests.get(url, stream=True, timeout=120, headers=headers) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    temporary.replace(destination)


def extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def download_gas(force: bool = False) -> None:
    destination = PROJECT_ROOT / "data" / "raw" / "gas_sensor"
    if list(destination.rglob("batch*.dat")) and not force:
        print("Gas Sensor dataset already exists; skipping download.")
        return
    archive = destination / "gas_sensor_drift.zip"
    download(GAS_URL, archive)
    extract(archive, destination)
    print(f"Gas Sensor dataset extracted to {destination}")


def download_skab(force: bool = False) -> None:
    destination = PROJECT_ROOT / "data" / "raw" / "skab"
    if list(destination.rglob("*.csv")) and not force:
        print("SKAB dataset already exists; skipping download.")
        return
    archive = destination / "skab_master.zip"
    download(SKAB_URL, archive)
    extract(archive, destination)
    print(f"SKAB dataset extracted to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets used by this project.")
    parser.add_argument("--gas", action="store_true", help="Download the UCI Gas Sensor dataset")
    parser.add_argument("--skab", action="store_true", help="Download SKAB")
    parser.add_argument("--all", action="store_true", help="Download both datasets")
    parser.add_argument("--force", action="store_true", help="Download even when data files already exist")
    arguments = parser.parse_args()
    if not (arguments.gas or arguments.skab or arguments.all):
        parser.error("Choose --gas, --skab, or --all")
    if arguments.all or arguments.gas:
        download_gas(arguments.force)
    if arguments.all or arguments.skab:
        download_skab(arguments.force)


if __name__ == "__main__":
    main()
