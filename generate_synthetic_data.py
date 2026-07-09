"""
generate_synthetic_data.py
==========================
FALLBACK ONLY - Generates synthetic data when no internet connection or
real GIRO data is available (unit testing, CI, offline development).

For real research runs, use:
    python fetch_real_data.py

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Synthetic station definitions (lat, lon)
# ---------------------------------------------------------------------------
SYNTHETIC_STATIONS = [
    ("DB049", -17.8,  25.9),
    ("EG931",  30.1,  31.4),
    ("JR055",  43.8,  12.9),
    ("AT138", -51.7, -57.9),
    ("WI937",  37.9, -75.5),
]


def diurnal_tec(lt: float, doy: int, lat: float) -> float:
    """Generate a physically plausible TEC value (TECU)."""
    diurnal  = 0.5 * (1 + np.sin(2 * np.pi * (lt - 6) / 24))
    seasonal = 0.5 * (1 + np.sin(2 * np.pi * (doy - 80) / 365.25))
    lat_fac  = np.cos(np.radians(lat)) ** 2
    base     = 20.0 + 40.0 * diurnal * (0.5 + 0.5 * seasonal) * (0.5 + 0.5 * lat_fac)
    return float(np.clip(base + np.random.normal(0, 2), 3.0, 100.0))


def generate_synthetic_dataset(
    n_stations: int = 5,
    n_days: int = 365,
    resolution_minutes: int = 15,
    output_path: str = "data/raw/ionosonde_data.csv",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic ionosonde CSV dataset.

    Parameters
    ----------
    n_stations : int
        Number of stations to simulate.
    n_days : int
        Number of days to simulate.
    resolution_minutes : int
        Temporal resolution in minutes.
    output_path : str
        Output CSV path.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated synthetic DataFrame (pipeline-schema compatible).
    """
    print("NOTE: Generating SYNTHETIC data for testing only.")
    print("      For real research, use: python fetch_real_data.py\n")

    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    stations = SYNTHETIC_STATIONS[:n_stations]
    start_date = pd.Timestamp("2020-01-01 00:00:00")
    freq = f"{resolution_minutes}min"
    timestamps = pd.date_range(
        start_date,
        periods=n_days * 24 * (60 // resolution_minutes),
        freq=freq,
    )

    records = []
    for station_id, lat, lon in stations:
        print(f"  Generating station: {station_id} ({lat:.1f}, {lon:.1f}) ...")
        for ts in timestamps:
            lt  = (ts.hour + ts.minute / 60 + lon / 15.0) % 24
            doy = ts.day_of_year

            tec      = diurnal_tec(lt, doy, lat)
            fof2     = float(np.clip(0.9 * np.sqrt(tec / 1.24e-5) * 1e-3 + rng.normal(0, 0.3), 1.5, 18.0))
            hmf2     = float(np.clip(250 + 80 * np.sin(2 * np.pi * (lt - 14) / 24) + rng.normal(0, 15), 150, 500))
            b0       = float(np.clip(80 + 40 * np.random.rand() + rng.normal(0, 5), 30, 250))
            b1       = float(np.clip(3.0 + rng.normal(0, 0.3), 1.5, 6.0))
            scale_f2 = float(np.clip(hmf2 * 0.4 + rng.normal(0, 8), 20, 180))
            zhalf    = float(np.clip(hmf2 - 30 + rng.normal(0, 10), 80, 450))
            yf2      = float(np.clip(b0 * 0.6, 10, 200))
            ff       = float(np.clip(rng.normal(1.0, 0.1), 0.5, 2.0))
            qf       = int(np.random.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.10, 0.05]))

            f107     = float(np.clip(100 + 80 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 10), 65, 300))
            kp       = float(np.clip(abs(rng.normal(1.5, 1.2)), 0, 8))
            ap       = float(np.clip(5 + 5 * kp + rng.normal(0, 2), 0, 300))
            dst      = float(np.clip(rng.normal(-10, 15), -200, 30))
            ssn      = float(np.clip(60 + 100 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 10), 0, 250))

            topside_fraction = 0.35 + 0.20 * np.sin(2 * np.pi * (lt - 14) / 24)
            topside_tec = float(np.clip(tec * topside_fraction + rng.normal(0, 1.5), 0.5, 60.0))

            records.append({
                "Timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Station":    station_id,
                "Latitude":   lat,
                "Longitude":  lon,
                "TEC":        round(tec, 4),
                "foF2":       round(fof2, 3),
                "hmF2":       round(hmf2, 2),
                "scaleF2":    round(scale_f2, 2),
                "B0":         round(b0, 2),
                "B1":         round(b1, 3),
                "zhalfNm":    round(zhalf, 2),
                "yF2":        round(yf2, 2),
                "FF":         round(ff, 4),
                "QF":         qf,
                "F10.7":      round(f107, 2),
                "Kp":         round(kp, 1),
                "Ap":         round(ap, 1),
                "Dst":        round(dst, 1),
                "SSN":        round(ssn, 1),
                "DayOfYear":  doy,
                "LocalTime":  round(lt, 4),
                "TopsideTEC": round(topside_tec, 4),
            })

    df = pd.DataFrame(records)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nSynthetic dataset saved -> {out}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
    print(f"  Stations: {df['Station'].nunique()}")
    return df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate SYNTHETIC ionosonde data (fallback only). "
            "For real data use: python fetch_real_data.py"
        )
    )
    parser.add_argument("--stations",   type=int, default=5,   help="Number of synthetic stations.")
    parser.add_argument("--days",       type=int, default=365, help="Number of days to simulate.")
    parser.add_argument("--resolution", type=int, default=15,  help="Resolution in minutes.")
    parser.add_argument("--output",     type=str, default="data/raw/ionosonde_data.csv")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_synthetic_dataset(
        n_stations=args.stations,
        n_days=args.days,
        resolution_minutes=args.resolution,
        output_path=args.output,
        random_seed=args.seed,
    )
