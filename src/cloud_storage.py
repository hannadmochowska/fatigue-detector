"""
Cloud storage for classified_runs.csv.

When GITHUB_TOKEN + GITHUB_REPO are set (i.e. running on Streamlit Community
Cloud), the CSV is committed back to the repo after every save so it survives
container restarts.

When running locally these env vars are absent, so it falls back to plain
local-file I/O — no behaviour change for local development.

Required env vars / Streamlit secrets for cloud deployment:
    GITHUB_TOKEN       Personal Access Token with "repo" (write) scope
    GITHUB_REPO        e.g. "hanna/fatigue-detector"
    GITHUB_BRANCH      branch to commit to (default: "main")
    GITHUB_FILE_PATH   path inside repo (default: "src/classified_runs.csv")
"""
from __future__ import annotations

import base64
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
_TOKEN     = lambda: os.environ.get("GITHUB_TOKEN", "")
_REPO      = lambda: os.environ.get("GITHUB_REPO", "")
_BRANCH    = lambda: os.environ.get("GITHUB_BRANCH", "main")
_FILE_PATH = lambda: os.environ.get("GITHUB_FILE_PATH", "src/classified_runs.csv")

_CLASSIFIED_CSV = Path(__file__).parent / "classified_runs.csv"


def _cloud_mode() -> bool:
    return bool(_TOKEN() and _REPO())


def _api_headers() -> dict:
    return {
        "Authorization": f"token {_TOKEN()}",
        "Accept": "application/vnd.github.v3+json",
    }


def _api_url() -> str:
    return f"https://api.github.com/repos/{_REPO()}/contents/{_FILE_PATH()}"


def _get_remote_sha_and_df() -> tuple[str | None, pd.DataFrame]:
    """Fetch the current file SHA and content from GitHub."""
    try:
        resp = requests.get(
            _api_url(),
            headers=_api_headers(),
            params={"ref": _BRANCH()},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = base64.b64decode(data["content"]).decode("utf-8")
            sha = data["sha"]
            df = pd.read_csv(StringIO(content)) if content.strip() else pd.DataFrame()
            return sha, df
        elif resp.status_code == 404:
            return None, pd.DataFrame()
        resp.raise_for_status()
    except Exception:
        pass
    return None, pd.DataFrame()


# ── Public API ────────────────────────────────────────────────────────────────

def load_runs() -> pd.DataFrame:
    """
    Load classified_runs from the appropriate source.
    Cloud: fetches from GitHub (always latest committed state).
    Local: reads local CSV.
    """
    if _cloud_mode():
        _, df = _get_remote_sha_and_df()
        # Also write to local path so run_pipeline.py can read it
        if not df.empty:
            df.to_csv(_CLASSIFIED_CSV, index=False)
        return df
    else:
        if _CLASSIFIED_CSV.exists():
            try:
                return pd.read_csv(_CLASSIFIED_CSV)
            except Exception:
                pass
        return pd.DataFrame()


def push_runs(df: pd.DataFrame | None = None, message: str = "Update classified runs") -> bool:
    """
    Commit the current classified_runs.csv to GitHub.
    If df is provided, writes it to the local file first.
    Returns True on success, False otherwise.
    """
    if not _cloud_mode():
        return True  # nothing to push in local mode

    # Write df to local file if provided
    if df is not None:
        df.to_csv(_CLASSIFIED_CSV, index=False)

    # Read from local file to commit
    if not _CLASSIFIED_CSV.exists():
        return False
    try:
        content_bytes = _CLASSIFIED_CSV.read_bytes()
        encoded = base64.b64encode(content_bytes).decode("utf-8")
        sha, _ = _get_remote_sha_and_df()

        payload: dict = {
            "message": message,
            "content": encoded,
            "branch": _BRANCH(),
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(
            _api_url(),
            json=payload,
            headers=_api_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False


def ensure_local_up_to_date() -> None:
    """
    Pull the latest classified_runs.csv from GitHub to the local container.
    Call once at app startup in cloud mode so run_pipeline.py reads fresh data.
    """
    if _cloud_mode():
        load_runs()  # side-effect: writes to local CSV
