"""
fetch_real_data.py
==================
Real-time data fetcher for the Mamba-TKAN TEC Reconstruction pipeline.

Sources
-------
1. **GIRO Digisonde** (giro.uml.edu)
   Fetches ionosonde parameters: foF2, hmF2, zhalfNm, yF2, scaleF2,
   B0, B1, TEC, FF, QF for one or more URSI-coded stations.

2. **OMNIWeb** (omniweb.gsfc.nasa.gov)
   Fetches hourly space weather indices: Kp, Ap, Dst, F10.7, SSN.

Output
------
A single merged CSV compatible with the Phase 1 pipeline schema:

    Timestamp, Station, Latitude, Longitude, TEC, foF2, hmF2,
    scaleF2, B0, B1, zhalfNm, yF2, FF, QF, F10.7, Kp, Ap,
    Dst, SSN, DayOfYear, LocalTime, TopsideTEC

Usage
-----
    python fetch_real_data.py
    python fetch_real_data.py --output data/raw/ionosonde_data.csv
"""

from __future__ import annotations

import argparse
import calendar
import io
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: make project root importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Known GIRO station catalogue  (URSI code -> (latitude, longitude, name))
# ---------------------------------------------------------------------------
GIRO_STATION_CATALOGUE: Dict[str, Tuple[float, float, str]] = {
    "DB049": (-17.83,  25.90, "Dimawe, Botswana"),
    "OU427": (-25.88,  28.19, "Olifantsfontein, South Africa"),
    "TN105": ( 36.83,  10.23, "Tunis, Tunisia"),
    "EG931": ( 30.08,  31.29, "Cairo, Egypt"),
    "ET854": (  8.75,  38.68, "Addis Ababa, Ethiopia"),
    "JR055": ( 43.78,  12.92, "San Benedetto, Italy"),
    "RL052": ( 51.56,   4.40, "Dourbes, Belgium"),
    "PQ052": ( 50.00,  14.50, "Prague, Czech Republic"),
    "AT138": ( 37.90,  23.50, "Athens, Greece"),
    "DK654": ( 55.47,  25.48, "Juliusruh, Germany"),
    "WI937": ( 37.93, -75.47, "Wallops Island, USA"),
    "BC840": ( 52.24,-106.53, "Saskatoon, Canada"),
    "RL753": ( 18.46, -66.15, "Ramey, Puerto Rico"),
    "LL721": ( 21.38,-157.98, "Lualualei, Hawaii"),
    "SM760": (-51.70, -57.90, "Stanley, Falklands"),
    "BR840": (-15.77, -47.92, "Brasilia, Brazil"),
    "TO536": ( 35.71, 139.49, "Tokyo, Japan"),
    "CH918": ( 31.10, 121.20, "Shanghai, China"),
    "WU430": ( 30.53, 114.35, "Wuhan, China"),
    "IN918": ( 12.97,  77.59, "Bengaluru, India"),
    "HY418": ( 37.40, 127.00, "Hwaseong, South Korea"),
    "AS00Q": (-35.30, 149.00, "Canberra, Australia"),
    "TW637": ( 25.04, 121.51, "Zhongli, Taiwan"),
}

GIRO_PARAM_MAP: Dict[str, str] = {
    "foF2":    "0",
    "hmF2":    "17",
    "zhalfNm": "20",
    "yF2":     "21",
    "scaleF2": "24",
    "B0":      "25",
    "B1":      "26",
    "TEC":     "28",
    "FF":      "29",
    "QF":      "31",
}

DESIRED_PARAMS = ["foF2", "hmF2", "zhalfNm", "yF2", "scaleF2", "B0", "B1", "TEC", "FF", "QF"]

# ---------------------------------------------------------------------------
# FIXED: Updated OMNIWeb variable codes matching your reference implementation
# ---------------------------------------------------------------------------
OMNIWEB_VAR_CODES = [
    ("38", "Kp"),
    ("39", "SSN"),
    ("40", "Dst"),
    ("49", "Ap"),
    ("50", "F10.7"),
]


# =============================================================================
# GIRO Fetcher
# =============================================================================

class GIROFetcher:
    BASE_URL = "https://giro.uml.edu/didbase/scaled.php"

    def fetch(
        self,
        station: str,
        year: int,
        month: int,
        day_start: int,
        day_end: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        _, last_day_of_month = calendar.monthrange(year, month)
        if day_end is None:
            day_end = last_day_of_month

        date_start = f"{year}-{month:02d}-{day_start:02d} 00:00"
        date_end = f"{year}-{month:02d}-{day_end:02d} 23:59"

        print(f"  [GIRO] Fetching {station}  {date_start} -> {date_end} ...", end="", flush=True)

        params = [
            ("location", station),
            ("date_start", date_start),
            ("date_end", date_end),
            ("query_submit", "Search"),
        ]
        for code in GIRO_PARAM_MAP.values():
            params.append(("chosenchars[]", code))

        encoded = urllib.parse.urlencode(params).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL,
            data=encoded,
            headers={"User-Agent": "Mozilla/5.0 (research pipeline)"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8")
        except Exception as exc:
            print(f" FAILED ({exc})")
            return None

        return self._parse_html(html, station)

    @staticmethod
    def _parse_html(html: str, station: str) -> Optional[pd.DataFrame]:
        lines = html.splitlines()
        header_line = None
        data_lines = []

        for line in lines:
            if line.startswith("# Time"):
                header_line = line
            elif not line.startswith("#") and len(line.strip()) > 0:
                data_lines.append(line)

        if not header_line or not data_lines:
            print(" no data found.")
            return None

        raw_cols = header_line.lstrip("#").strip().split()
        col_names: List[str] = []
        for i, c in enumerate(raw_cols):
            if c == "QD":
                col_names.append(f"{raw_cols[i - 1]}_QD")
            else:
                col_names.append(c)

        records = []
        for line in data_lines:
            parts = line.split()
            row = dict(zip(col_names, parts))
            records.append(row)

        df = pd.DataFrame(records)
        print(f" {len(df)} records.")

        if df.empty:
            return None

        df["Timestamp"] = pd.to_datetime(df["Time"], errors="coerce")
        df.drop(columns=["Time"], inplace=True, errors="ignore")

        keep_cols = ["Timestamp"] + [p for p in DESIRED_PARAMS if p in df.columns]
        df = df[keep_cols].copy()

        for col in DESIRED_PARAMS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Station"] = station
        return df


# =============================================================================
# OMNIWeb Fetcher
# =============================================================================

class OMNIWebFetcher:
    BASE_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

    def fetch(self, year: int, month: int) -> Optional[pd.DataFrame]:
        _, last_day = calendar.monthrange(year, month)
        start = f"{year:04d}{month:02d}01"
        end = f"{year:04d}{month:02d}{last_day:02d}"

        print(f"  [OMNIWeb] Fetching space weather {start} -> {end} ...", end="", flush=True)

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        form_data = [
            ("activity", "ftp"),
            ("res", "hour"),
            ("spacecraft", "omni2"),
            ("start_date", start),
            ("end_date", end),
        ]
        for code, _ in OMNIWEB_VAR_CODES:
            form_data.append(("vars", code))

        encoded = urllib.parse.urlencode(form_data).encode("utf-8")
        req = urllib.request.Request(self.BASE_URL, data=encoded)

        try:
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                html = resp.read().decode("utf-8")
        except Exception as exc:
            print(f" form submission FAILED ({exc})")
            return None

        lst_match = re.search(
            r'href="(https://omniweb\.gsfc\.nasa\.gov/staging/omni2_[a-zA-Z0-9_]+\.lst)"', html
        )
        fmt_match = re.search(
            r'href="(https://omniweb\.gsfc\.nasa\.gov/staging/omni2_[a-zA-Z0-9_]+\.fmt)"', html
        )

        if not lst_match or not fmt_match:
            print(" could not find data links.")
            return None

        lst_url = lst_match.group(1)
        fmt_url = fmt_match.group(1)

        try:
            with urllib.request.urlopen(fmt_url, context=ctx, timeout=30) as resp:
                fmt_content = resp.read().decode("utf-8")
            headers = self._parse_fmt_headers(fmt_content)
        except Exception:
            headers = ["YEAR", "DOY", "Hour", "Kp", "SSN", "Dst", "Ap", "F10.7"]

        try:
            with urllib.request.urlopen(lst_url, context=ctx, timeout=60) as resp:
                lst_content = resp.read().decode("utf-8")
        except Exception as exc:
            print(f" data download FAILED ({exc})")
            return None

        df = pd.read_csv(io.StringIO(lst_content), sep=r"\s+", header=None, names=headers)

        # -----------------------------------------------------------------
        # Robust substring mapping to handle dynamic NASA header variations
        # -----------------------------------------------------------------
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "kp" in col_lower:
                rename_map[col] = "Kp"
            elif "sunspot" in col_lower or "r (" in col_lower or "r sunspot" in col_lower:
                rename_map[col] = "SSN"
            elif "dst" in col_lower:
                rename_map[col] = "Dst"
            elif "ap" in col_lower:
                rename_map[col] = "Ap"
            elif "f10.7" in col_lower or "solar index" in col_lower:
                rename_map[col] = "F10.7"
            elif "year" in col_lower:
                rename_map[col] = "YEAR"
            elif "doy" in col_lower or "day" in col_lower:
                rename_map[col] = "DOY"
            elif "hour" in col_lower or "hr" in col_lower:
                rename_map[col] = "Hour"

        df.rename(columns=rename_map, inplace=True)

        if {"YEAR", "DOY", "Hour"}.issubset(df.columns):
            df["Timestamp"] = pd.to_datetime(
                df["YEAR"].astype(str) + " " + df["DOY"].astype(str),
                format="%Y %j",
                errors="coerce",
            ) + pd.to_timedelta(df["Hour"], unit="h")
            df.drop(columns=["YEAR", "DOY", "Hour"], inplace=True, errors="ignore")

        # Data Cleaning & Scale Adjustments
        SW_COLS = ["Kp", "SSN", "Dst", "Ap", "F10.7"]
        FILL_VALUES = {999, 9999, 9999.0, 99999, 999.9, 99.9}
        KP_FILL_RAW = 99   
        
        for col in SW_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if col == "Kp":
                    # Kp is delivered scaled as Kp*10, convert back to standard decimal scale
                    df[col] = df[col].apply(lambda v: np.nan if v == KP_FILL_RAW else v / 10.0)
                elif col == "Ap":
                    df[col] = df[col].apply(lambda v: np.nan if v in {999, 9999, 9999.0} else v)
                else:
                    df[col] = df[col].apply(lambda v: np.nan if v in FILL_VALUES else v)

        keep = ["Timestamp"] + [c for c in SW_COLS if c in df.columns]
        df = df[keep]

        print(f" {len(df)} hourly records.")
        return df

    @staticmethod
    def _parse_fmt_headers(fmt_content: str) -> List[str]:
        headers = []
        for line in fmt_content.split("\n"):
            line = line.strip()
            m = re.match(r"^\s*\d+\s+(.+?)\s+[A-ZIF][0-9.]+\s*$", line)
            if m:
                headers.append(m.group(1).strip())
        return headers if headers else ["YEAR", "DOY", "Hour", "Kp", "SSN", "Dst", "Ap", "F10.7"]


# =============================================================================
# Coordinate & Merger System Mechanics
# =============================================================================

def get_station_coords(station: str) -> Tuple[float, float]:
    if station in GIRO_STATION_CATALOGUE:
        lat, lon, name = GIRO_STATION_CATALOGUE[station]
        print(f"    Station {station}: {name}  ({lat:.2f} N, {lon:.2f} E)")
        return lat, lon

    print(f"    Station '{station}' not in catalogue. Please enter coordinates:")
    lat = float(input(f"      Latitude  for {station} (deg, N positive): ").strip())
    lon = float(input(f"      Longitude for {station} (deg, E positive): ").strip())
    GIRO_STATION_CATALOGUE[station] = (lat, lon, "User-defined")
    return lat, lon


def build_pipeline_dataframe(
    giro_df: pd.DataFrame,
    omni_df: Optional[pd.DataFrame],
    station: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    df = giro_df.copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].dt.tz is not None:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if omni_df is not None and not omni_df.empty:
        omni = omni_df.copy()
        omni["Timestamp"] = pd.to_datetime(omni["Timestamp"], errors="coerce")
        if omni["Timestamp"].dt.tz is not None:
            omni["Timestamp"] = omni["Timestamp"].dt.tz_localize(None)
        omni = omni.dropna(subset=["Timestamp"])
        omni = omni.sort_values("Timestamp").reset_index(drop=True)

        df["_merge_ts"] = df["Timestamp"].dt.floor("h")
        omni_key = omni.rename(columns={"Timestamp": "_merge_ts"})

        df = pd.merge(df, omni_key, on="_merge_ts", how="left")
        df.drop(columns=["_merge_ts"], inplace=True)
    else:
        for col in ["Kp", "SSN", "Dst", "Ap", "F10.7"]:
            df[col] = np.nan

    df["Station"] = station
    df["Latitude"] = lat
    df["Longitude"] = lon
    df["DayOfYear"] = df["Timestamp"].dt.day_of_year.astype("float32")

    utc_decimal_hour = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    df["LocalTime"] = (utc_decimal_hour + lon / 15.0) % 24.0
    df["TopsideTEC"] = np.nan

    SCHEMA = [
        "Timestamp", "Station", "Latitude", "Longitude",
        "TEC", "foF2", "hmF2", "scaleF2", "B0", "B1",
        "zhalfNm", "yF2", "FF", "QF",
        "F10.7", "Kp", "Ap", "Dst", "SSN",
        "DayOfYear", "LocalTime", "TopsideTEC",
    ]
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    return df[SCHEMA]


def fetch_real_data(
    stations: List[str],
    year: int,
    month: int,
    day_start: int = 1,
    day_end: Optional[int] = None,
    output_path: str = "data/raw/ionosonde_data.csv",
) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  Mamba-TKAN TEC Pipeline - Real-Time Data Fetcher")
    print("=" * 65)

    giro = GIROFetcher()
    omni = OMNIWebFetcher()

    print("\n[1/3] Fetching OMNIWeb space weather ...")
    omni_df = omni.fetch(year, month)

    print(f"\n[2/3] Fetching GIRO ionosonde data ({len(stations)} station(s)) ...")
    all_frames: List[pd.DataFrame] = []

    for stn in stations:
        print(f"\n  Station: {stn}")
        try:
            lat, lon = get_station_coords(stn)
        except Exception as exc:
            print(f"    Skipping {stn}: coordinate error ({exc})")
            continue

        giro_df = giro.fetch(stn, year, month, day_start, day_end)
        if giro_df is None or giro_df.empty:
            print(f"    No GIRO data for {stn}. Skipping.")
            continue

        merged = build_pipeline_dataframe(giro_df, omni_df, stn, lat, lon)
        all_frames.append(merged)
        print(f"    Built {len(merged)} rows for {stn}.")

    if not all_frames:
        print("\nNo data fetched for any station.")
        return pd.DataFrame()

    print(f"\n[3/3] Saving merged dataset ...")
    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.sort_values(["Station", "Timestamp"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out, index=False, float_format="%.6f", date_format="%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 65)
    print(f"  Dataset saved -> {out.resolve()}")
    print(f"  Total rows    : {len(full_df)}")
    print(f"  Stations      : {full_df['Station'].nunique()}")
    print(f"  Date range    : {full_df['Timestamp'].min()} -> {full_df['Timestamp'].max()}")
    print(f"  TEC mean      : {full_df['TEC'].mean():.3f} TECU")
    print(f"  TopsideTEC    : {full_df['TopsideTEC'].isna().sum()} NaN values")
    print("=" * 65)

    return full_df


def interactive_mode() -> None:
    print("=" * 65)
    print("  GIRO + OMNIWeb Real-Time Data Downloader")
    print("=" * 65)

    year_str = input("\nEnter Year (e.g., 2024): ").strip()
    month_str = input("Enter Month (1-12): ").strip()
    day_start_str = input("Enter Start Day (1-31, default 1): ").strip() or "1"
    day_end_str = input("Enter End Day (1-31, blank = end of month): ").strip() or ""

    try:
        year = int(year_str)
        month = int(month_str)
        day_start = int(day_start_str)
        day_end = int(day_end_str) if day_end_str else None
    except ValueError as exc:
        print(f"Invalid input: {exc}")
        sys.exit(1)

    print("\nEnter GIRO station URSI codes (comma-separated).")
    station_input = input("Station(s): ").strip().upper()
    stations = [s.strip() for s in station_input.split(",") if s.strip()]
    if not stations:
        sys.exit(1)

    output_path = input("\nOutput CSV path [data/raw/ionosonde_data.csv]: ").strip() or "data/raw/ionosonde_data.csv"
    fetch_real_data(stations, year, month, day_start, day_end, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch real GIRO + OMNIWeb data.")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day-start", type=int, default=1)
    parser.add_argument("--day-end", type=int, default=None)
    parser.add_argument("--stations", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/raw/ionosonde_data.csv")
    parser.add_argument("--list-stations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_stations:
        for code, (lat, lon, name) in sorted(GIRO_STATION_CATALOGUE.items()):
            print(f"{code:<10} {lat:>7.2f} {lon:>8.2f}  {name}")
        sys.exit(0)

    if args.year and args.month and args.stations:
        stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
        fetch_real_data(stations, args.year, args.month, args.day_start, args.day_end, args.output)
    else:
        interactive_mode()