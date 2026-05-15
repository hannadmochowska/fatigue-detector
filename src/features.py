import numpy as np
import pandas as pd
from parse_tcx import parse_tcx

# --- utility helpers -----------------------------------------------------

def _safe_cv(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mean()
    v = s.std(ddof=0)
    if m == 0 or np.isnan(m) or np.isnan(v):
        return np.nan
    return v / m


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

    df_1s = df[cols].resample("1s").mean()

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

def compute_micro_pause_features(df_1s: pd.DataFrame) -> dict:
    """
    Detect short 'micro-pauses' (1–5 s) on flat terrain where speed drops
    sharply but doesn't turn into a full long stop.

    Returns:
        micro_pause_count
        micro_pause_per_km
        micro_pause_mean_s
    """
    feats = {
        "micro_pause_count": np.nan,
        "micro_pause_per_km": np.nan,
        "micro_pause_mean_s": np.nan,
    }

    if "speed_mps" not in df_1s.columns or "distance_m" not in df_1s.columns:
        return feats

    df = df_1s.copy()
    # Use only flat seconds for this metric
    if "is_flat" in df.columns:
        df = df[df["is_flat"]]

    if df.empty:
        return feats

    # Moving median speed (exclude standing still)
    moving = df["speed_mps"][df["speed_mps"] > 0.5]
    if moving.empty:
        return feats

    median_speed = moving.median()
    if median_speed <= 0:
        return feats

    # A 'pause' = speed < 30% of median running speed
    df["is_pause"] = df["speed_mps"] < 0.3 * median_speed

    # Label contiguous pause segments
    pause_indices = np.where(df["is_pause"].values)[0]
    if len(pause_indices) == 0:
        feats["micro_pause_count"] = 0
        feats["micro_pause_per_km"] = 0.0
        feats["micro_pause_mean_s"] = 0.0
        return feats

    segments = []
    start = pause_indices[0]
    prev = pause_indices[0]

    for idx in pause_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx
    segments.append((start, prev))

    durations = []
    micro_count = 0

    for s_idx, e_idx in segments:
        dur = e_idx - s_idx + 1  # seconds
        if 1 <= dur <= 5:
            micro_count += 1
            durations.append(dur)
        # >5 s treated as full stop (e.g. red light) → ignored

    if micro_count == 0:
        feats["micro_pause_count"] = 0
        feats["micro_pause_per_km"] = 0.0
        feats["micro_pause_mean_s"] = 0.0
        return feats

    # distance in km
    dist_m = df_1s["distance_m"].iloc[-1] - df_1s["distance_m"].iloc[0]
    dist_km = dist_m / 1000.0 if dist_m > 0 else np.nan

    feats["micro_pause_count"] = micro_count
    feats["micro_pause_mean_s"] = float(np.mean(durations))
    feats["micro_pause_per_km"] = (
        micro_count / dist_km if dist_km and not np.isnan(dist_km) else np.nan
    )
    return feats


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

def compute_warmup_features(df_1s: pd.DataFrame) -> dict:
    """
    Estimate how long it takes HR to stabilise.
    We look at first 20 min (or full run if shorter), smooth HR,
    and find the first time where HR slope stays small for ~3 min.
    """
    feats = {"warmup_stab_time_s": np.nan}

    if "hr" not in df_1s.columns or "elapsed_s" not in df_1s.columns:
        return feats

    df = df_1s.copy()
    max_window_s = min(df["elapsed_s"].iloc[-1], 20 * 60)  # cap at 20 min
    df = df[df["elapsed_s"] <= max_window_s]

    if len(df) < 60:
        return feats

    # Smooth HR with 30 s rolling mean
    hr_smooth = df["hr"].rolling(window=30, min_periods=15).mean()

    # Approximate slope over 30 s steps
    slope = (hr_smooth.diff(30) / 30).abs()  # bpm per second

    # We consider HR 'stable' if |slope| < 0.05 bpm/s (~3 bpm/min)
    stable = slope < 0.05

    # Require stability for at least 3 minutes
    stable_run = stable.rolling(window=180, min_periods=60).mean()

    stable_indices = np.where(stable_run >= 0.9)[0]
    if len(stable_indices) == 0:
        # never really stabilised → treat warmup_stab_time as full early window
        feats["warmup_stab_time_s"] = float(max_window_s)
    else:
        t_stab = df["elapsed_s"].iloc[stable_indices[0]]
        feats["warmup_stab_time_s"] = float(t_stab)

    return feats

def compute_finish_collapse_features(df_1s: pd.DataFrame) -> dict:
    """
    Compare pace in mid-run vs final 10% of the run as an 'end collapse' metric.
    Positive % means slowdown in the final part.
    """
    feats = {"finish_collapse_pct": np.nan}

    if "pace_s_per_km" not in df_1s.columns or "elapsed_s" not in df_1s.columns:
        return feats

    df = df_1s.copy()
    if len(df) < 120:
        return feats

    total_time = df["elapsed_s"].iloc[-1]
    if total_time <= 0:
        return feats

    # Mid-run: 40–60% of time
    mid_start = 0.4 * total_time
    mid_end = 0.6 * total_time
    mid = df[(df["elapsed_s"] >= mid_start) & (df["elapsed_s"] <= mid_end)]

    # Final: last 10% of time
    final_start = 0.9 * total_time
    final = df[df["elapsed_s"] >= final_start]

    if len(mid) < 30 or len(final) < 30:
        return feats

    mid_pace = mid["pace_s_per_km"].mean()
    final_pace = final["pace_s_per_km"].mean()
    if mid_pace <= 0:
        return feats

    collapse_pct = (final_pace - mid_pace) / mid_pace * 100.0
    feats["finish_collapse_pct"] = float(collapse_pct)
    return feats

def compute_hr_recovery_features(df_1s: pd.DataFrame,
                                 window_s: int = 10,
                                 threshold_frac: float = 0.08) -> dict:
    """
    For each pace surge, measure how much HR drops 10 s after peak.
    Higher recovery = better autonomic response.
    """
    feats = {"hr_recovery_10s_mean": np.nan}

    if "speed_mps" not in df_1s.columns or "hr" not in df_1s.columns:
        return feats

    speeds = df_1s["speed_mps"].values
    hrs = df_1s["hr"].values
    n = len(df_1s)
    if n <= window_s + 10:
        return feats

    surge_indices = []
    i = window_s
    while i < n - 10:
        prev = speeds[i - window_s]
        curr = speeds[i]
        if prev > 0:
            change = (curr - prev) / prev
            if change > threshold_frac:
                surge_indices.append(i)
                # skip ahead to avoid counting the same surge repeatedly
                i += window_s
                continue
        i += 1

    if not surge_indices:
        return feats

    recoveries = []
    for idx in surge_indices:
        hr_peak = hrs[idx]
        hr_after = hrs[idx + 10]
        recoveries.append(hr_peak - hr_after)

    if recoveries:
        feats["hr_recovery_10s_mean"] = float(np.mean(recoveries))
    return feats

def compute_short_hr_drift_features(df_1s: pd.DataFrame) -> dict:
    """
    Short-timescale HR variability: average 60 s rolling std of HR.
    Higher values = more jitter / unstable control.
    """
    feats = {"hr_std_60s_mean": np.nan}

    if "hr" not in df_1s.columns:
        return feats

    if len(df_1s) < 60:
        return feats

    hr_roll_std = df_1s["hr"].rolling(window=60, min_periods=30).std()
    feats["hr_std_60s_mean"] = float(hr_roll_std.mean())
    return feats

def compute_session_summary_features(df_1s: pd.DataFrame) -> dict:
    """
    Basic per-run summary features for session-type detection.
    """
    feats = {
        "duration_min": np.nan,
        "total_distance_km": np.nan,
        "mean_hr": np.nan,
        "pace_cv_run": np.nan,
    }

    if "elapsed_s" not in df_1s.columns or "distance_m" not in df_1s.columns:
        return feats

    duration_s = df_1s["elapsed_s"].iloc[-1]
    feats["duration_min"] = float(duration_s / 60.0)

    dist_m = df_1s["distance_m"].iloc[-1] - df_1s["distance_m"].iloc[0]
    feats["total_distance_km"] = float(dist_m / 1000.0) if dist_m > 0 else np.nan

    if "hr" in df_1s.columns:
        feats["mean_hr"] = float(df_1s["hr"].mean())

    if "pace_s_per_km" in df_1s.columns:
        feats["pace_cv_run"] = _safe_cv(df_1s["pace_s_per_km"])

    return feats

def compute_neuro_features(df_1s: pd.DataFrame) -> dict:
    """
    Bundle all neuro / cognitive-control related metrics into a single call.
    """
    feats = {}
    feats.update(compute_micro_pause_features(df_1s))
    feats.update(compute_warmup_features(df_1s))
    feats.update(compute_finish_collapse_features(df_1s))
    feats.update(compute_hr_recovery_features(df_1s))
    feats.update(compute_short_hr_drift_features(df_1s))
    feats.update(compute_session_summary_features(df_1s))
    return feats

import numpy as np
import pandas as pd

def compute_interval_pattern_features(df_1s: pd.DataFrame) -> dict:
    """
    Detect whether the run contains *structured* intervals:
    - repeated fast "ON" segments
    - each ON segment at least ~20 s
    - with slower "OFF" segments between them
    - ON durations and gaps between ONs reasonably consistent

    Returns:
        interval_on_count           – number of detected ON segments (>= 20 s)
        interval_off_count          – number of detected OFF segments
        interval_patterns_consistent – True if pattern looks like real intervals
    """
    feats = {
        "interval_on_count": 0,
        "interval_off_count": 0,
        "interval_patterns_consistent": False,
    }

    if "speed_mps" not in df_1s.columns:
        return feats

    df = df_1s.copy()

    # 1 Hz data → smooth speed over ~7 s window to remove noise
    df["speed_smooth"] = df["speed_mps"].rolling(7, center=True, min_periods=1).median()

    # Use relative thresholds: fast vs base pace
    base_speed = df["speed_smooth"].median()
    if not np.isfinite(base_speed) or base_speed <= 0:
        return feats

    fast_thresh = base_speed * 1.15   # ≈ >15% faster than median pace
    slow_thresh = base_speed * 0.90   # ≈ clearly slower than median

    df["is_fast"] = df["speed_smooth"] > fast_thresh
    df["is_slow"] = df["speed_smooth"] < slow_thresh

    n = len(df)
    if n == 0:
        return feats

    # Helper: collect contiguous segments from a boolean mask
    def collect_segments(mask):
        segments = []
        start = None
        for i, flag in enumerate(mask):
            if flag:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i - 1))
                    start = None
        if start is not None:
            segments.append((start, n - 1))
        return segments

    fast_segments_all = collect_segments(df["is_fast"].values)
    slow_segments_all = collect_segments(df["is_slow"].values)

    # Only keep "real" ON segments: at least 20 s
    min_on_dur = 20      # seconds
    min_off_dur = 10     # seconds

    on_segments = [(s, e) for (s, e) in fast_segments_all
                   if (e - s + 1) >= min_on_dur]
    off_segments = [(s, e) for (s, e) in slow_segments_all
                    if (e - s + 1) >= min_off_dur]

    on_durs = [e - s + 1 for (s, e) in on_segments]
    off_durs = [e - s + 1 for (s, e) in off_segments]

    feats["interval_on_count"] = len(on_durs)
    feats["interval_off_count"] = len(off_durs)

    # Need at least 3 ON segments to call it "intervals"
    if len(on_durs) < 3:
        return feats

    # Coefficient of variation (allowing more variability than before)
    def cv(arr):
        if len(arr) == 0:
            return np.nan
        m = np.mean(arr)
        return np.std(arr) / m if m > 0 else np.nan

    on_cv = cv(on_durs)

    # Gaps between ON segments (time from end of one ON to start of next)
    gaps = []
    for (s1, e1), (s2, e2) in zip(on_segments[:-1], on_segments[1:]):
        gap = max(0, s2 - e1 - 1)
        gaps.append(gap)

    gaps_cv = cv(gaps)

    # We are generous here: real workouts are not perfect metronomes.
    # Typical criteria:
    # - ON durations within ~40% CV
    # - gaps between ONs within ~50% CV
    consistent_on = (not np.isnan(on_cv)) and (on_cv <= 0.40)
    consistent_gaps = (len(gaps) >= 2) and (not np.isnan(gaps_cv)) and (gaps_cv <= 0.50)

    if consistent_on and consistent_gaps:
        feats["interval_patterns_consistent"] = True

    return feats

# -------------------------------------------------------------
# CML / GPV SCORE AGGREGATION
# -------------------------------------------------------------


def compute_cml_gpv_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CML / GPV fatigue scores to the per-run feature table.

    Uses existing columns from earlier feature functions:
      - hr_mean_drift, late_hr_mean, pace_s_per_km_mean_drift
      - flat_hr_mean_drift, flat_pace_mean_drift, is_hilly_run
      - pace_cv_early/late, cadence_cv_early/late
      - surges_all_10s_8pct, surges_late_10s_8pct
      - finish_collapse_pct, warmup_stab_time_s, micro_pause_per_km
      - hr_std_60s_mean, hr_recovery_10s_mean
    """

    df = features_df.copy()

    # ---------------------------------------------------------
    # 1. Ratios from existing features
    # ---------------------------------------------------------
    df["pace_cv_ratio"] = df["pace_cv_late"] / df["pace_cv_early"]
    df["cadence_cv_ratio"] = df["cadence_cv_late"] / df["cadence_cv_early"]
    df["surge_ratio"] = df["surges_late_10s_8pct"] / df["surges_all_10s_8pct"]

    for col in ["pace_cv_ratio", "cadence_cv_ratio", "surge_ratio"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------
    # 2. Helper: z-score that works even with few runs
    # ---------------------------------------------------------
    def zscore(series: pd.Series) -> pd.Series:
        # if everything is NaN → treat as neutral (0)
        if series.notna().sum() == 0:
            return pd.Series(0.0, index=series.index)

        m = series.mean()
        s = series.std(ddof=0)

        # single run / zero variance → neutral (0)
        if np.isnan(s) or s == 0:
            return pd.Series(0.0, index=series.index)

        return (series - m) / s

    # ---------------------------------------------------------
    # 3. Physiology / terrain-based z-scores
    # ---------------------------------------------------------
    # Full-run drifts
    df["z_hr_mean_drift_full"] = zscore(df["hr_mean_drift"])
    df["z_late_hr_mean"] = zscore(df["late_hr_mean"])
    df["z_pace_mean_drift_full"] = zscore(df["pace_s_per_km_mean_drift"])

    # Flat-only drifts
    df["z_hr_mean_drift_flat"] = zscore(df["flat_hr_mean_drift"])
    df["z_pace_mean_drift_flat"] = zscore(df["flat_pace_mean_drift"])

    # Variability / surges
    df["z_pace_cv_ratio"] = zscore(df["pace_cv_ratio"])
    df["z_cadence_cv_ratio"] = zscore(df["cadence_cv_ratio"])
    df["z_surge_ratio"] = zscore(df["surge_ratio"])

    # ---------------------------------------------------------
    # 4. Physiology-based CML and GPV
    # ---------------------------------------------------------
    df["CML_full_z"] = (
        df["z_hr_mean_drift_full"]
        + 0.5 * df["z_late_hr_mean"]
        + 0.5 * df["z_pace_mean_drift_full"]
    )

    df["CML_flat_z"] = (
        df["z_hr_mean_drift_flat"]
        + 0.5 * df["z_pace_mean_drift_flat"]
    )

    df["CML_phys_z"] = np.where(
        df["is_hilly_run"] & df["CML_flat_z"].notna(),
        df["CML_flat_z"],
        df["CML_full_z"],
    )

    df["GPV_phys_z"] = (
        df["z_pace_cv_ratio"]
        + 0.5 * df["z_cadence_cv_ratio"]
        + 0.5 * df["z_surge_ratio"]
    )

    # ---------------------------------------------------------
    # 5. Neuro-ish contributions (warmup, collapse, recovery)
    # ---------------------------------------------------------
    df["z_micro_pause_per_km"] = zscore(df["micro_pause_per_km"])
    df["z_warmup_stab_time_s"] = zscore(df["warmup_stab_time_s"])
    df["z_finish_collapse_pct"] = zscore(df["finish_collapse_pct"])
    df["z_hr_std_60s_mean"] = zscore(df["hr_std_60s_mean"])
    # Invert recovery: lower recovery = worse → higher z = worse
    df["z_hr_recovery_deficit"] = zscore(-df["hr_recovery_10s_mean"])

    df["CML_neuro_z"] = (
        0.4 * df["z_finish_collapse_pct"]
        + 0.3 * df["z_warmup_stab_time_s"]
        + 0.3 * df["z_hr_std_60s_mean"]
    )

    df["GPV_neuro_z"] = (
        0.4 * df["z_micro_pause_per_km"]
        + 0.3 * df["z_hr_std_60s_mean"]
        + 0.3 * df["z_hr_recovery_deficit"]
    )

    # ---------------------------------------------------------
    # 6. Final combined scores
    # ---------------------------------------------------------
    df["CML_z"] = (df["CML_phys_z"] + df["CML_neuro_z"]).fillna(0.0)
    df["GPV_z"] = (df["GPV_phys_z"] + df["GPV_neuro_z"]).fillna(0.0)

    return df