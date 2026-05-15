"""
Garmin Connect integration for Fatigue Detector.
Mirrors the gc-strava project approach: garminconnect + .env credentials.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TOKENS_DIR = str(Path(__file__).parent / ".garmin_tokens")


def _get_client(email: str | None = None, password: str | None = None):
    from garminconnect import Garmin

    email = email or os.environ.get("GARMIN_EMAIL", "")
    password = password or os.environ.get("GARMIN_PASSWORD", "")
    if not email or not password:
        raise ValueError(
            "GARMIN_EMAIL and GARMIN_PASSWORD must be set in .env "
            "or passed directly."
        )
    client = Garmin(email, password)
    client.login(tokenstore=TOKENS_DIR)
    return client


def get_recent_runs(
    email: str | None = None,
    password: str | None = None,
    days: int = 30,
) -> list[dict]:
    """Return recent running activities as simplified dicts."""
    client = _get_client(email, password)
    end = date.today()
    start = end - timedelta(days=days)

    try:
        activities = client.get_activities_by_date(
            start.isoformat(), end.isoformat(), activitytype="running"
        )
    except Exception:
        # fall back to all activities and filter manually
        activities = client.get_activities_by_date(
            start.isoformat(), end.isoformat()
        )

    if not isinstance(activities, list):
        return []

    runs = []
    for a in activities:
        atype = (a.get("activityType") or {}).get("typeKey", "").lower()
        if "run" not in atype and "treadmill" not in atype:
            continue
        dist = a.get("distance") or 0
        runs.append(
            {
                "id": a.get("activityId"),
                "name": a.get("activityName", "Run"),
                "date": (a.get("startTimeLocal") or a.get("startTimeGMT", ""))[:10],
                "distance_km": round(dist / 1000, 2),
                "duration_min": round((a.get("duration") or 0) / 60, 1),
                "type": atype,
                "source": "garmin",
            }
        )
    return runs


def download_tcx(
    email: str | None = None,
    password: str | None = None,
    activity_id: int = 0,
) -> str:
    """
    Download the TCX file for activity_id.
    Writes to a temp file and returns the path.
    Caller is responsible for deleting the file after use.
    """
    client = _get_client(email, password)
    tcx_bytes = client.download_activity(
        activity_id,
        dl_fmt=client.ActivityDownloadFormat.TCX,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".tcx", delete=False)
    tmp.write(tcx_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name
