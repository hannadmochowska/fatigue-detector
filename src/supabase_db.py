"""
Supabase database layer for multi-user Fatigue Detector.

Required Streamlit secrets / env vars:
    SUPABASE_URL          https://your-project.supabase.co
    SUPABASE_ANON_KEY     Public anon key  (used for auth sign-in/sign-up)
    SUPABASE_SERVICE_KEY  Service role key (used for all DB reads/writes)
    ENCRYPTION_KEY        Fernet key for encrypting stored Garmin/Strava passwords
                          Generate once with:
                            python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from __future__ import annotations

import json
import os

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
from supabase import create_client, Client


# ── Config ────────────────────────────────────────────────────────────────────
def _url() -> str:        return os.environ.get("SUPABASE_URL", "")
def _anon() -> str:       return os.environ.get("SUPABASE_ANON_KEY", "")
def _service() -> str:    return os.environ.get("SUPABASE_SERVICE_KEY", "")
def _enc_key() -> str:    return os.environ.get("ENCRYPTION_KEY", "")

def is_enabled() -> bool:
    return bool(_url() and _anon() and _service())

def _anon_client() -> Client:
    return create_client(_url(), _anon())

def _db() -> Client:
    return create_client(_url(), _service())


# ── Encryption ────────────────────────────────────────────────────────────────
def _cipher() -> Fernet | None:
    key = _enc_key()
    if not key:
        return None
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        return None

def encrypt(value: str) -> str:
    if not value:
        return ""
    c = _cipher()
    return c.encrypt(value.encode()).decode() if c else value

def decrypt(value: str) -> str:
    if not value:
        return ""
    c = _cipher()
    if not c:
        return value
    try:
        return c.decrypt(value.encode()).decode()
    except (InvalidToken, Exception):
        return value


# ── Auth ──────────────────────────────────────────────────────────────────────
def sign_up(email: str, password: str) -> None:
    resp = _anon_client().auth.sign_up({"email": email, "password": password})
    if resp.user is None:
        raise ValueError("Sign-up failed — check your email to confirm your account.")

def sign_in(email: str, password: str) -> tuple[str, str]:
    """Returns (user_id, access_token)."""
    resp = _anon_client().auth.sign_in_with_password({"email": email, "password": password})
    if resp.user is None:
        raise ValueError("Incorrect email or password.")
    return resp.user.id, resp.session.access_token


# ── Classified runs ───────────────────────────────────────────────────────────
def load_runs(user_id: str) -> pd.DataFrame:
    try:
        rows = _db().table("classified_runs").select("*").eq("user_id", user_id).execute().data
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.rename(columns={"cml_z": "CML_z", "gpv_z": "GPV_z",
                                   "cml_adj": "CML_adj", "gpv_adj": "GPV_adj"})
    except Exception:
        return pd.DataFrame()

def save_run(user_id: str, run_data: dict) -> None:
    try:
        _db().table("classified_runs").upsert({
            "user_id":     user_id,
            "run_id":      run_data.get("run_id"),
            "date":        run_data.get("date"),
            "session_type":run_data.get("session_type"),
            "neuro_tag":   run_data.get("neuro_tag"),
            "final_label": run_data.get("final_label"),
            "base_label":  run_data.get("base_label"),
            "final_code":  run_data.get("final_code"),
            "cml_z":       run_data.get("CML_z"),
            "gpv_z":       run_data.get("GPV_z"),
            "cml_adj":     run_data.get("CML_adj"),
            "gpv_adj":     run_data.get("GPV_adj"),
            "reasons":     run_data.get("reasons"),
        }, on_conflict="user_id,run_id").execute()
    except Exception:
        pass

def delete_run(user_id: str, run_id: str) -> None:
    try:
        _db().table("classified_runs").delete().eq("user_id", user_id).eq("run_id", run_id).execute()
    except Exception:
        pass


# ── Features history (for z-score computation) ────────────────────────────────
def load_features(user_id: str) -> pd.DataFrame:
    try:
        rows = _db().table("features_history").select("*").eq("user_id", user_id).execute().data
        if not rows:
            return pd.DataFrame()
        result = []
        for r in rows:
            feats = r.get("features", {})
            if isinstance(feats, str):
                feats = json.loads(feats)
            result.append({"run_id": r["run_id"],
                           "session_type": r.get("session_type"),
                           "neuro_tag": r.get("neuro_tag"),
                           **feats})
        return pd.DataFrame(result)
    except Exception:
        return pd.DataFrame()

def save_features(user_id: str, run_id: str,
                  session_type: str | None, neuro_tag: str | None,
                  features: dict) -> None:
    try:
        feat_cols = {k: v for k, v in features.items()
                     if k not in ("run_id", "session_type", "neuro_tag", "user_id")}
        _db().table("features_history").upsert({
            "user_id":      user_id,
            "run_id":       run_id,
            "session_type": session_type,
            "neuro_tag":    neuro_tag,
            "features":     json.dumps({k: (float(v) if hasattr(v, "item") else v)
                                        for k, v in feat_cols.items()}),
        }, on_conflict="user_id,run_id").execute()
    except Exception:
        pass


# ── User integrations (Garmin / Strava credentials) ───────────────────────────
def load_integrations(user_id: str) -> dict:
    try:
        resp = _db().table("user_integrations").select("*").eq("user_id", user_id).execute()
        if resp.data:
            d = resp.data[0]
            return {
                "garmin_email":          d.get("garmin_email", ""),
                "garmin_password":       decrypt(d.get("garmin_password_enc", "")),
                "strava_client_id":      d.get("strava_client_id", ""),
                "strava_client_secret":  decrypt(d.get("strava_client_secret_enc", "")),
                "strava_refresh_token":  decrypt(d.get("strava_refresh_token_enc", "")),
            }
    except Exception:
        pass
    return {}

def save_integrations(user_id: str, integrations: dict) -> None:
    try:
        _db().table("user_integrations").upsert({
            "user_id":                  user_id,
            "garmin_email":             integrations.get("garmin_email", ""),
            "garmin_password_enc":      encrypt(integrations.get("garmin_password", "")),
            "strava_client_id":         integrations.get("strava_client_id", ""),
            "strava_client_secret_enc": encrypt(integrations.get("strava_client_secret", "")),
            "strava_refresh_token_enc": encrypt(integrations.get("strava_refresh_token", "")),
        }, on_conflict="user_id").execute()
    except Exception:
        pass
