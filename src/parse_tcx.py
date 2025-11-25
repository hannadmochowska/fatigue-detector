from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def parse_tcx(file_path: str | Path) -> pd.DataFrame:
    """
    Parse a Garmin TCX file into a DataFrame with columns:

        time (datetime, also used as index)
        hr
        cadence
        distance_m
        speed_mps
        pace_s_per_km
        elev_m
        elapsed_s

    The function is robust to missing cadence/speed by trying multiple
    possible tag locations and falling back to distance-derived speed.
    """
    file_path = Path(file_path)

    # TCX / Garmin namespaces
    ns = {
        "tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
        "ns3": "http://www.garmin.com/xmlschemas/ActivityExtension/v2",
    }

    tree = ET.parse(file_path)
    root = tree.getroot()

    rows: list[dict] = []

    for tp in root.findall(".//tcx:Trackpoint", ns):
        # ---- time ----
        t_text = tp.findtext("tcx:Time", namespaces=ns)
        if t_text is None:
            continue
        t = pd.to_datetime(t_text)

        # ---- distance ----
        dist_txt = tp.findtext("tcx:DistanceMeters", namespaces=ns)
        distance_m = float(dist_txt) if dist_txt is not None else np.nan

        # ---- heart rate ----
        hr_el = tp.find("tcx:HeartRateBpm/tcx:Value", ns)
        hr = float(hr_el.text) if hr_el is not None else np.nan

        # ---- altitude ----
        alt_txt = tp.findtext("tcx:AltitudeMeters", namespaces=ns)
        elev_m = float(alt_txt) if alt_txt is not None else np.nan

        # ---- extensions: speed + cadence ----
        speed_mps = np.nan
        cadence = np.nan

        ext = tp.find("tcx:Extensions", ns)
        if ext is not None:
            # try Garmin ActivityExtension TPX node, but allow any namespace
            tpx = ext.find(".//ns3:TPX", ns) or ext.find(".//{*}TPX")
            if tpx is not None:
                # Speed
                s_txt = (
                    tpx.findtext(".//ns3:Speed", namespaces=ns)
                    or tpx.findtext(".//{*}Speed")
                )
                if s_txt is not None:
                    try:
                        speed_mps = float(s_txt)
                    except ValueError:
                        speed_mps = np.nan

                # Cadence (RunCadence, Cadence, any Cadence tag)
                c_txt = (
                    tpx.findtext(".//ns3:RunCadence", namespaces=ns)
                    or tpx.findtext(".//ns3:Cadence", namespaces=ns)
                    or tpx.findtext(".//{*}Cadence")
                )
                if c_txt is not None:
                    try:
                        cadence = float(c_txt)
                    except ValueError:
                        cadence = np.nan

        # Fallback: any Cadence directly under Trackpoint if still NaN
        if np.isnan(cadence):
            c_txt = (
                tp.findtext("tcx:Cadence", namespaces=ns)
                or tp.findtext(".//{*}Cadence")
            )
            if c_txt is not None:
                try:
                    cadence = float(c_txt)
                except ValueError:
                    cadence = np.nan

        rows.append(
            {
                "time": t,
                "hr": hr,
                "cadence": cadence,
                "distance_m": distance_m,
                "speed_mps": speed_mps,
                "elev_m": elev_m,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "time",
                "hr",
                "cadence",
                "distance_m",
                "speed_mps",
                "elev_m",
                "elapsed_s",
                "pace_s_per_km",
            ]
        )

    df = pd.DataFrame(rows)

    # Sort by time and use as index
    df = df.sort_values("time").reset_index(drop=True)
    df = df.set_index("time", drop=False)

    # Elapsed seconds since start
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    # Forward-fill distance (Garmin often repeats last value)
    if df["distance_m"].notna().any():
        df["distance_m"] = df["distance_m"].ffill()

    # If speed is mostly missing, derive from distance / time
    speed_missing_ratio = df["speed_mps"].isna().mean()
    if speed_missing_ratio > 0.2:
        ddist = df["distance_m"].diff()
        dt = df["elapsed_s"].diff()
        # avoid division by zero
        dt = dt.replace(0, np.nan)
        derived_speed = ddist / dt
        df["speed_mps"] = df["speed_mps"].fillna(derived_speed)

    # Pace in seconds per km (protect against zeros)
    df["speed_mps"] = df["speed_mps"].replace(0, np.nan)
    df["pace_s_per_km"] = 1000.0 / df["speed_mps"]

    return df
