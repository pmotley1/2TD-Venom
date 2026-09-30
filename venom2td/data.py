"""Download and cache nflverse data releases."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
CACHE = Path(__file__).resolve().parent.parent / ".cache"


def _fetch(path: str, refresh: bool) -> Path:
    dest = CACHE / path.replace("/", "__")
    if refresh or not dest.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{path}", dest)
    return dest


def pbp(season: int, refresh: bool = False) -> pd.DataFrame:
    return pd.read_parquet(_fetch(f"pbp/play_by_play_{season}.parquet", refresh))


def schedule(refresh: bool = False) -> pd.DataFrame:
    return pd.read_csv(_fetch("schedules/games.csv", refresh))


def rosters(season: int, refresh: bool = False) -> pd.DataFrame:
    return pd.read_parquet(_fetch(f"weekly_rosters/roster_weekly_{season}.parquet", refresh))


def injuries(season: int, refresh: bool = False) -> pd.DataFrame:
    return pd.read_parquet(_fetch(f"injuries/injuries_{season}.parquet", refresh))


def snaps(season: int, refresh: bool = False) -> pd.DataFrame:
    return pd.read_parquet(_fetch(f"snap_counts/snap_counts_{season}.parquet", refresh))
