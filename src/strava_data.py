"""
Strava integration for Fatigue Detector.
Same credentials pattern as gc-strava: CLIENT_ID / CLIENT_SECRET / REFRESH_TOKEN in .env.
One-time auth: run strava_auth.py from the gc-strava project to get your refresh token.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

STRAVA_API_BASE = "https://www.strava.com/api/v3"
TOKEN_URL = "https://www.strava.com/oauth/token"

# Strava running cadence is reported as strides/min (one stride = 2 steps).
# Multiply by 2 to get steps/min, matching Garmin's cadence field.
STRAVA_CADENCE_FACTOR = 2


def get_access_token() -> str:
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": os.environ["STRAVA_CLIENT_ID"],
            "client_secret": os.environ["STRAVA_CLIENT_SECRET"],
            "refresh_token": os.environ["STRAVA_REFRESH_TOKEN"],
            "grant_type": "refresh_token",
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _headers() -> dict:
    return {"Authorization": f"Bearer {get_access_token()}"}


def get_recent_runs(days: int = 30) -> list[dict]:
    """Return recent running activities as simplified dicts."""
    after_ts = int(datetime.now(timezone.utc).timestamp()) - (days * 86400)
    resp = requests.get(
        f"{STRAVA_API_BASE}/athlete/activities",
        headers=_headers(),
        params={"per_page": 100, "after": after_ts},
    )
    resp.raise_for_status()

    runs = []
    for a in resp.json():
        if a.get("type", "").lower() not in ("run", "virtualrun"):
            continue
        runs.append(
            {
                "id": a["id"],
                "name": a.get("name", "Run"),
                "date": a.get("start_date_local", "")[:10],
                "distance_km": round(a.get("distance", 0) / 1000, 2),
                "duration_min": round(a.get("moving_time", 0) / 60, 1),
                "type": "run",
                "source": "strava",
            }
        )
    return runs


def get_activity_dataframe(activity_id: int) -> pd.DataFrame:
    """
    Fetch Strava activity streams and return a DataFrame in the same format
    as parse_tcx():
        time, hr, cadence, distance_m, speed_mps, elev_m, elapsed_s, pace_s_per_km
    """
    # 1. Fetch streams
    streams_resp = requests.get(
        f"{STRAVA_API_BASE}/activities/{activity_id}/streams",
        headers=_headers(),
        params={
            "keys": "time,distance,heartrate,cadence,velocity_smooth,altitude",
            "key_by_type": "true",
        },
    )
    streams_resp.raise_for_status()
    streams = streams_resp.json()

    def _data(key):
        s = streams.get(key, {})
        return s.get("data") if isinstance(s, dict) else None

    time_data = _data("time")
    if not time_data:
        raise ValueError(f"No time stream available for Strava activity {activity_id}")

    n = len(time_data)
    distance = _data("distance") or [np.nan] * n
    hr = _data("heartrate") or [np.nan] * n
    cadence_raw = _data("cadence") or [np.nan] * n
    velocity = _data("velocity_smooth") or [np.nan] * n
    altitude = _data("altitude") or [np.nan] * n

    # 2. Get activity start time for absolute timestamps
    act_resp = requests.get(
        f"{STRAVA_API_BASE}/activities/{activity_id}", headers=_headers()
    )
    act_resp.raise_for_status()
    act = act_resp.json()
    start_time = pd.to_datetime(act.get("start_date_local", "2000-01-01T00:00:00"))

    # 3. Build DataFrame
    cadence_spm = [
        c * STRAVA_CADENCE_FACTOR if c is not None and not np.isnan(c) else np.nan
        for c in cadence_raw
    ]

    df = pd.DataFrame(
        {
            "elapsed_s": time_data,
            "distance_m": distance,
            "hr": hr,
            "cadence": cadence_spm,
            "speed_mps": velocity,
            "elev_m": altitude,
        }
    )

    # Absolute time index (matches parse_tcx output)
    df["time"] = start_time + pd.to_timedelta(df["elapsed_s"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    df = df.set_index("time", drop=False)

    # Pace
    df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce").replace(0, np.nan)
    df["pace_s_per_km"] = 1000.0 / df["speed_mps"]

    return df
