"""
End-to-end fatigue pipeline: raw DataFrame → classification result.

Usage (command line):
    python run_pipeline.py activity.tcx
    python run_pipeline.py ./tcx_folder/ --session-type easy
    python run_pipeline.py activity.tcx --neuro-tag body_limited

Also importable from app.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from classifier import classify_run
from features import (
    compute_cml_gpv_scores,
    compute_early_late_features,
    compute_interval_pattern_features,
    compute_neuro_features,
    compute_surge_features,
    compute_terrain_features,
    resample_to_1s,
)
from parse_tcx import parse_tcx

# History file — reuses classified_runs_v1.csv if it already exists,
# otherwise starts fresh at features_history.csv
_SRC = Path(__file__).parent
HISTORY_CSV = _SRC / "classified_runs_v1.csv" if (_SRC / "classified_runs_v1.csv").exists() else _SRC / "features_history.csv"
CLASSIFIED_CSV = _SRC / "classified_runs.csv"


def compute_all_features(df_1s: pd.DataFrame) -> dict:
    """Run every feature function and merge into a single flat dict."""
    feats: dict = {}
    feats.update(compute_early_late_features(df_1s))
    feats.update(compute_surge_features(df_1s))
    feats.update(compute_terrain_features(df_1s))
    feats.update(compute_neuro_features(df_1s))
    feats.update(compute_interval_pattern_features(df_1s))
    return feats


def _resolve_date(run_date: str | None, df_1s: pd.DataFrame) -> str:
    """Return a YYYY-MM-DD string from the explicit date arg or the DataFrame index."""
    # 1. Prefer explicitly supplied date (from Strava/Garmin API metadata)
    if run_date:
        try:
            ts = pd.to_datetime(run_date)
            if ts.year >= 2000:
                return str(ts.date())
        except Exception:
            pass
    # 2. Fall back to the first timestamp in the resampled DataFrame
    if len(df_1s) > 0:
        try:
            ts = pd.to_datetime(df_1s.index[0])
            if ts.year >= 2000:
                return str(ts.date())
        except Exception:
            pass
    return ""


def run_single(
    df_raw: pd.DataFrame,
    run_id: str = "run",
    neuro_tag: str | None = None,
    session_type: str | None = None,
    save_history: bool = True,
    run_date: str | None = None,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    """
    Full pipeline for a single run.

    Returns:
        result      – classify_run output dict (includes CML_z, GPV_z, final_label, reasons …)
        scored_row  – pd.Series with all features + z-scores for this run
        df_1s       – 1 Hz resampled DataFrame (for plotting)
    """
    df_1s = resample_to_1s(df_raw)
    feats = compute_all_features(df_1s)

    # Load existing history and append new run
    if HISTORY_CSV.exists():
        history = pd.read_csv(HISTORY_CSV, low_memory=False)
    else:
        history = pd.DataFrame()

    new_row = pd.DataFrame(
        [{"run_id": run_id, "session_type": session_type, "neuro_tag": neuro_tag, **feats}]
    )

    # Drop duplicate run_id before appending (re-analysis of same run)
    if "run_id" in history.columns:
        history = history[history["run_id"] != run_id]
    elif "file" in history.columns:
        # legacy column name
        history = history[history["file"] != run_id]

    combined = pd.concat([history, new_row], ignore_index=True)

    # Compute z-scores relative to all runs in history
    scored = compute_cml_gpv_scores(combined)
    this_run = scored.iloc[-1]

    # Classify
    result = classify_run(
        cml_z=float(this_run["CML_z"]),
        gpv_z=float(this_run["GPV_z"]),
        neuro_tag=neuro_tag,
        session_type=session_type,
    )
    result["CML_z"] = float(this_run["CML_z"])
    result["GPV_z"] = float(this_run["GPV_z"])

    if save_history:
        # Persist updated history (features only, not z-scores — they shift as runs are added)
        new_row_with_meta = new_row.copy()
        combined_features = pd.concat(
            [history[[c for c in history.columns if c in new_row.columns]],
             new_row_with_meta],
            ignore_index=True,
        )
        combined_features.to_csv(HISTORY_CSV, index=False)

        # Persist classification result
        classified_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "date": _resolve_date(run_date, df_1s),
                    "session_type": session_type,
                    "neuro_tag": neuro_tag,
                    "final_label": result["final_label"],
                    "base_label": result["base_label"],
                    "final_code": result["final_code"],
                    "CML_z": result["CML_z"],
                    "GPV_z": result["GPV_z"],
                    "CML_adj": result["CML_adj"],
                    "GPV_adj": result["GPV_adj"],
                    "reasons": result["reasons"],
                }
            ]
        )
        if CLASSIFIED_CSV.exists():
            existing = pd.read_csv(CLASSIFIED_CSV)
            existing = existing[existing["run_id"] != run_id]
            updated = pd.concat([existing, classified_row], ignore_index=True)
        else:
            updated = classified_row
        updated.to_csv(CLASSIFIED_CSV, index=False)

        # Push to GitHub when running on Streamlit Community Cloud
        try:
            from cloud_storage import push_runs
            push_runs(message=f"Analyse run {run_id}")
        except Exception:
            pass

    return result, this_run, df_1s


def run_folder(
    folder: str,
    session_type: str | None = None,
    neuro_tag: str | None = None,
) -> list[dict]:
    """Batch-process all TCX files in a folder."""
    folder = Path(folder)
    tcx_files = sorted(folder.glob("*.tcx"))
    print(f"Found {len(tcx_files)} TCX files in {folder}")

    results = []
    for f in tcx_files:
        print(f"Processing {f.name}...", end=" ", flush=True)
        try:
            df_raw = parse_tcx(f)
            result, _, _ = run_single(
                df_raw, run_id=f.name, session_type=session_type, neuro_tag=neuro_tag
            )
            print(
                f"→ {result['final_label']} "
                f"(CML={result['CML_z']:.2f}, GPV={result['GPV_z']:.2f})"
            )
            results.append({"file": f.name, **result})
        except Exception as e:
            print(f"ERROR: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fatigue Detector pipeline")
    parser.add_argument("input", help="TCX file or folder of TCX files")
    parser.add_argument(
        "--session-type",
        default=None,
        choices=["easy", "long", "intervals", "race", "tempo"],
    )
    parser.add_argument(
        "--neuro-tag",
        default=None,
        choices=["body_limited", "regulation_limited", "global_fatigue", "fresh"],
    )
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_dir():
        run_folder(str(p), session_type=args.session_type, neuro_tag=args.neuro_tag)
    else:
        df_raw = parse_tcx(p)
        result, _, _ = run_single(
            df_raw,
            run_id=p.name,
            neuro_tag=args.neuro_tag,
            session_type=args.session_type,
        )
        print(f"\nResult:  {result['final_label']}")
        print(f"CML_z:   {result['CML_z']:.2f}")
        print(f"GPV_z:   {result['GPV_z']:.2f}")
        print(f"Reasons: {result['reasons']}")
