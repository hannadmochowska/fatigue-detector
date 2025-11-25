import numpy as np
import pandas as pd
from parse_tcx import parse_tcx


import numpy as np
import pandas as pd

def resample_to_1s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample raw data to 1 Hz and compute terrain-related signals.

    Input df must contain:
        time, speed_mps, pace_s_per_km, hr, cadence, distance_m, elev_m
    """
    df = df.copy()
    df = df.set_index("time")

    # columns to resample with mean + interpolate
    cols = ["speed_mps", "pace_s_per_km", "hr", "cadence", "distance_m", "elev_m"]

    df_1s = df[cols].resample("1S").mean()

    # interpolate through gaps
    for col in cols:
        df_1s[col] = df_1s[col].interpolate(limit_direction="both")

    # elapsed time in seconds from start
    df_1s["elapsed_s"] = (df_1s.index - df_1s.index[0]).total_seconds()

    # distance / elevation deltas
    df_1s["delta_dist_m"] = df_1s["distance_m"].diff()
    df_1s["delta_elev"] = df_1s["elev_m"].diff()

    # avoid tiny or negative distance deltas (GPS noise / standing still)
    df_1s.loc[df_1s["delta_dist_m"] < 0.5, "delta_dist_m"] = np.nan

    # grade (unitless), then as percent
    df_1s["grade"] = df_1s["delta_elev"] / df_1s["delta_dist_m"]
    df_1s["grade"] = df_1s["grade"].clip(-0.3, 0.3)  # clip insane spikes
    df_1s["grade_pct"] = df_1s["grade"] * 100.0

    # vertical speed
    # resampled at 1 Hz → delta_elev is already m/s
    df_1s["vertical_speed_mps"] = df_1s["delta_elev"]
    df_1s["vertical_speed_mh"] = df_1s["vertical_speed_mps"] * 3600.0

    # terrain flags
    uphill_thr = 0.02  # 2% grade
    df_1s["is_uphill"] = df_1s["grade"] > uphill_thr
    df_1s["is_downhill"] = df_1s["grade"] < -uphill_thr
    df_1s["is_flat"] = (~df_1s["is_uphill"]) & (~df_1s["is_downhill"])

    # convenience alias for early/late code
    df_1s["cadence_spm"] = df_1s["cadence"]

    return df_1s


def split_early_late(df: pd.DataFrame, frac: float = 0.3):
    """Return first frac and last frac of the run by elapsed time."""
    total = df["elapsed_s"].iloc[-1]
    early_cut = total * frac
    late_cut = total * (1 - frac)

    early_df = df[df["elapsed_s"] <= early_cut]
    late_df = df[df["elapsed_s"] >= late_cut]
    return early_df, late_df


def basic_stats(segment: pd.DataFrame, col: str) -> dict:
    s = segment[col].dropna()
    if len(s) == 0:
        return {"mean": np.nan, "std": np.nan, "cv": np.nan}
    mean = s.mean()
    std = s.std()
    cv = std / mean if mean != 0 else np.nan
    return {"mean": mean, "std": std, "cv": cv}


import numpy as np
import pandas as pd

def _cv(series: pd.Series) -> float:
    """Coefficient of variation (std/mean) with safe NaN handling."""
    series = series.dropna()
    if series.empty:
        return np.nan
    m = series.mean()
    s = series.std(ddof=0)
    if m == 0 or np.isnan(m) or np.isnan(s):
        return np.nan
    return s / m


def _cv(series: pd.Series) -> float:
    """Coefficient of variation (std/mean) with safe NaN handling."""
    series = series.dropna()
    if series.empty:
        return np.nan
    m = series.mean()
    s = series.std(ddof=0)
    if m == 0 or np.isnan(m) or np.isnan(s):
        return np.nan
    return s / m


def compute_early_late_features(df_1s: pd.DataFrame) -> dict:
    """
    Early/late HR, pace, and cadence features for fatigue detection.

    Expects df_1s to have columns:
        - 'hr'              (bpm)
        - 'pace_s_per_km'   (sec/km)
        - 'cadence_spm'     (steps per minute)
        - 'is_flat'         (bool)  # for flat-only drifts
    """
    feats = {}
    n = len(df_1s)
    if n < 10:
        # too short to say anything
        feats["early_hr_mean"] = np.nan
        feats["late_hr_mean"] = np.nan
        feats["hr_mean_drift"] = np.nan
        feats["early_pace_s_per_km_mean"] = np.nan
        feats["late_pace_s_per_km_mean"] = np.nan
        feats["pace_s_per_km_mean_drift"] = np.nan
        feats["pace_cv_early"] = np.nan
        feats["pace_cv_late"] = np.nan
        feats["pace_cv_ratio"] = np.nan
        feats["cadence_cv_early"] = np.nan
        feats["cadence_cv_late"] = np.nan
        feats["cadence_cv_ratio"] = np.nan
        feats["flat_hr_mean_drift"] = np.nan
        feats["flat_pace_mean_drift"] = np.nan
        return feats

    # --- full-run early/late segmentation ---
    early, late = split_early_late(df_1s, frac=0.3)

    # HR means + drift
    early_hr_mean = early["hr"].mean()
    late_hr_mean = late["hr"].mean()
    hr_mean_drift = late_hr_mean - early_hr_mean

    # Pace means + drift (positive = slowdown)
    early_pace_mean = early["pace_s_per_km"].mean()
    late_pace_mean = late["pace_s_per_km"].mean()
    pace_mean_drift = late_pace_mean - early_pace_mean

    # CVs & ratios
    pace_cv_early = _cv(early["pace_s_per_km"])
    pace_cv_late = _cv(late["pace_s_per_km"])
    pace_cv_ratio = (
        pace_cv_late / pace_cv_early
        if pace_cv_early not in (0, np.nan) else np.nan
    )

    cadence_cv_early = _cv(early["cadence_spm"])
    cadence_cv_late = _cv(late["cadence_spm"])
    cadence_cv_ratio = (
        cadence_cv_late / cadence_cv_early
        if cadence_cv_early not in (0, np.nan) else np.nan
    )

    feats.update({
        "early_hr_mean": early_hr_mean,
        "late_hr_mean": late_hr_mean,
        "hr_mean_drift": hr_mean_drift,
        "early_pace_s_per_km_mean": early_pace_mean,
        "late_pace_s_per_km_mean": late_pace_mean,
        "pace_s_per_km_mean_drift": pace_mean_drift,
        "pace_cv_early": pace_cv_early,
        "pace_cv_late": pace_cv_late,
        "pace_cv_ratio": pace_cv_ratio,
        "cadence_cv_early": cadence_cv_early,
        "cadence_cv_late": cadence_cv_late,
        "cadence_cv_ratio": cadence_cv_ratio,
    })

    # --- NEW: flat-only drift metrics ---
    if "is_flat" in df_1s.columns:
        flat = df_1s[df_1s["is_flat"]]

        # require at least ~2 minutes of flat running
        if len(flat) > 120:
            n_flat = len(flat)
            split_flat = n_flat // 2
            early_flat = flat.iloc[:split_flat]
            late_flat = flat.iloc[split_flat:]

            early_flat_hr_mean = early_flat["hr"].mean()
            late_flat_hr_mean = late_flat["hr"].mean()
            early_flat_pace_mean = early_flat["pace_s_per_km"].mean()
            late_flat_pace_mean = late_flat["pace_s_per_km"].mean()

            feats["flat_hr_mean_drift"] = late_flat_hr_mean - early_flat_hr_mean
            feats["flat_pace_mean_drift"] = (
                late_flat_pace_mean - early_flat_pace_mean
            )
        else:
            feats["flat_hr_mean_drift"] = np.nan
            feats["flat_pace_mean_drift"] = np.nan
    else:
        feats["flat_hr_mean_drift"] = np.nan
        feats["flat_pace_mean_drift"] = np.nan

    return feats


def count_pace_surges(
    df_1s: pd.DataFrame, window_s: int = 10, threshold_frac: float = 0.08
) -> int:
    """
    Count how many times speed changes by > threshold_frac within window_s.
    """
    s = df_1s["speed_mps"].dropna().reset_index(drop=True)
    n = len(s)
    if n <= window_s:
        return 0

    surges = 0
    for i in range(window_s, n):
        prev = s.iloc[i - window_s]
        curr = s.iloc[i]
        if prev <= 0:
            continue
        change = abs(curr - prev) / prev
        if change > threshold_frac:
            surges += 1
    return surges


def compute_surge_features(df_1s: pd.DataFrame) -> dict:
    early, late = split_early_late(df_1s, frac=0.3)
    feats = {}
    feats["surges_all_10s_8pct"] = count_pace_surges(df_1s, 10, 0.08)
    feats["surges_late_10s_8pct"] = count_pace_surges(late, 10, 0.08)
    return feats

def compute_terrain_features(df_1s: pd.DataFrame) -> dict:
    """
    Compute per-run terrain descriptors from 1 Hz dataframe.
    """
    feats = {}

    # total distance
    dist_start = df_1s["distance_m"].iloc[0]
    dist_end = df_1s["distance_m"].iloc[-1]
    total_dist_m = dist_end - dist_start
    total_dist_km = total_dist_m / 1000.0 if total_dist_m > 0 else np.nan

    # elevation gain (sum of positive deltas)
    delta_elev = df_1s["delta_elev"].fillna(0.0)
    total_elev_gain_m = delta_elev.clip(lower=0).sum()

    feats["total_elev_gain_m"] = float(total_elev_gain_m)
    feats["total_distance_km"] = float(total_dist_km)

    # elevation gain per km
    if not np.isnan(total_dist_km) and total_dist_km > 0:
        feats["elev_gain_per_km"] = total_elev_gain_m / total_dist_km
    else:
        feats["elev_gain_per_km"] = np.nan

    # time fractions
    n = len(df_1s)
    if n > 0:
        feats["pct_time_uphill"] = df_1s["is_uphill"].mean() * 100.0
        feats["pct_time_flat"] = df_1s["is_flat"].mean() * 100.0
        feats["pct_time_downhill"] = df_1s["is_downhill"].mean() * 100.0
    else:
        feats["pct_time_uphill"] = np.nan
        feats["pct_time_flat"] = np.nan
        feats["pct_time_downhill"] = np.nan

    # uphill-only stats
    uphill = df_1s[df_1s["is_uphill"]]

    if len(uphill) > 0:
        feats["mean_grade_uphill_pct"] = uphill["grade_pct"].mean()
        feats["mean_vspeed_uphill_mh"] = uphill["vertical_speed_mh"].mean()
    else:
        feats["mean_grade_uphill_pct"] = np.nan
        feats["mean_vspeed_uphill_mh"] = np.nan

    # simple "hilly" flag (tune threshold as needed)
    egpkm = feats["elev_gain_per_km"]
    if np.isnan(egpkm):
        feats["is_hilly_run"] = False
    else:
        feats["is_hilly_run"] = egpkm > 20.0  # e.g. >20 m gain per km

    return feats

