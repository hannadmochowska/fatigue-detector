"""
Fatigue Detector — Streamlit app.

Run with:
    cd src/
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from parse_tcx import parse_tcx
from run_pipeline import CLASSIFIED_CSV, run_single

load_dotenv()

# ── Password gate ─────────────────────────────────────────────────────────────
def _check_password() -> bool:
    correct = os.environ.get("APP_PASSWORD", "")
    if not correct:
        return True  # no password set → open access (local dev)
    if st.session_state.get("authenticated"):
        return True
    st.markdown("## 🏃 Fatigue Detector")
    pwd = st.text_input("Enter password", type="password", key="pwd_input")
    if st.button("Log in"):
        if pwd == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()
    return False

_check_password()

# (v1 migration removed — classified_runs_v1.csv only contains raw features, no dates or labels)

# ── Cloud storage: pull latest CSV from GitHub on each cold start ─────────────
try:
    from cloud_storage import ensure_local_up_to_date
    ensure_local_up_to_date()
except Exception:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fatigue Detector",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "A": "#e74c3c",
    "B": "#9b59b6",
    "C": "#e67e22",
    "D": "#27ae60",
    "M": "#95a5a6",
}
TYPE_LABELS = {
    "Type A — Body-limited":       "A",
    "Type B — Regulation-limited": "B",
    "Type C — Global fatigue":     "C",
    "Type D — Fresh / controlled": "D",
    "Mixed / normal":              "M",
}
COLOR_MAP = {
    "Type A — Body-limited":       TYPE_COLORS["A"],
    "Type B — Regulation-limited": TYPE_COLORS["B"],
    "Type C — Global fatigue":     TYPE_COLORS["C"],
    "Type D — Fresh / controlled": TYPE_COLORS["D"],
    "Mixed / normal":              TYPE_COLORS["M"],
}
NEURO_PHRASES: dict[str | None, str] = {
    None:            "— not set —",
    "legs_heavy":    "My legs felt heavy / muscles burning",
    "legs_lungs":    "Legs failed before lungs / HR felt fine but legs were done",
    "focus":         "Hard to focus / couldn't stay locked in",
    "mentally_fine": "Mentally drained but physically fine",
    "rhythm":        "Couldn't keep rhythm / pacing felt messy",
    "everything":    "Everything felt hard — tired mentally AND physically",
    "hr_pace":       "HR was high AND pacing fell apart",
    "smooth":        "Felt smooth and relaxed / easy rhythm",
    "could_go":      "Could have gone further / very low effort",
}
PHRASE_TO_TAG: dict[str | None, str | None] = {
    None: None, "legs_heavy": "body_limited", "legs_lungs": "body_limited",
    "focus": "regulation_limited", "mentally_fine": "regulation_limited",
    "rhythm": "regulation_limited", "everything": "global_fatigue",
    "hr_pace": "global_fatigue", "smooth": "fresh", "could_go": "fresh",
}
SESSION_OPTIONS: dict[str | None, str] = {
    None: "— auto detect —", "easy": "Easy", "long": "Long run",
    "intervals": "Intervals", "race": "Race", "tempo": "Tempo",
}
TYPE_EXPLANATIONS = {
    "A": "**High CML, Low GPV** — Strong physiological strain (HR drifted, pace slowed) but rhythm stayed controlled. Classic **peripheral / body-limited** fatigue.",
    "B": "**Low CML, High GPV** — Physiology calm but pacing became erratic or cadence unstable. Classic **central / regulation-limited** fatigue.",
    "C": "**High CML, High GPV** — Both physiological strain and pacing stability broke down. **Global fatigue**.",
    "D": "**Low CML, Low GPV** — Low strain, stable rhythm. You were **fresh and in control**.",
    "M": "**Mid-range CML and GPV** — No strong signal in either direction.",
}
TRAINING_TIPS = {
    "A": "**Training implication:** Body/peripheral fatigue dominated. Focus on fuelling, pacing strategy, and recovery.",
    "B": "**Training implication:** Central/cognitive fatigue dominated. Prioritise sleep, stress management, and mental load.",
    "C": "**Training implication:** Both systems flagged. Full recovery needed before the next hard session.",
    "D": "**Training implication:** You were fresh — good day for quality work.",
    "M": "**Training implication:** No strong fatigue signal either way.",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _patch_missing_dates(activities: list[dict]) -> None:
    """
    Remove rows with bad/missing dates from classified_runs.csv so the
    auto-analyser re-fetches them with correct dates and z-scores.
    """
    if not CLASSIFIED_CSV.exists() or not activities:
        return
    try:
        df = pd.read_csv(CLASSIFIED_CSV)
        if "date" not in df.columns:
            return

        def _is_bad_date(val) -> bool:
            try:
                ts = pd.to_datetime(val, errors="coerce")
                return pd.isna(ts) or ts.year < 2000
            except Exception:
                return True

        bad_mask = df["date"].apply(_is_bad_date)
        if bad_mask.any():
            df = df[~bad_mask]
            df.to_csv(CLASSIFIED_CSV, index=False)
    except Exception:
        pass


def _auto_analyse_all(
    activities: list[dict],
    source: str,
    g_email: str | None = None,
    g_password: str | None = None,
) -> int:
    """
    Silently run the pipeline on every activity not yet in classified_runs.csv.
    Uses session_type=None (auto-detect) and neuro_tag=None (neutral — no score impact).
    Returns the number of newly analysed runs.
    """
    # Build set of already-classified run_ids
    already_done: set[str] = set()
    if CLASSIFIED_CSV.exists():
        try:
            existing = pd.read_csv(CLASSIFIED_CSV)
            already_done = set(existing["run_id"].dropna().astype(str))
        except Exception:
            pass

    to_do = [a for a in activities if str(a["id"]) not in already_done]
    if not to_do:
        return 0

    bar = st.progress(0, text=f"Auto-analysing {len(to_do)} runs…")
    analysed = 0

    for i, act in enumerate(to_do):
        run_id = str(act["id"])
        try:
            if source == "strava":
                from strava_data import get_activity_dataframe
                df_raw = get_activity_dataframe(act["id"])
            elif source == "garmin":
                from garmin_data import download_tcx
                import os
                tmp_path = download_tcx(g_email, g_password, act["id"])
                from parse_tcx import parse_tcx as _parse_tcx
                df_raw = _parse_tcx(tmp_path)
                os.unlink(tmp_path)
            else:
                continue  # uploads are always manual

            run_single(
                df_raw,
                run_id=run_id,
                neuro_tag=None,
                session_type=None,
                run_date=act.get("date"),
                save_history=True,
            )
            analysed += 1
        except Exception:
            pass  # skip silently — bad data or API error for one run shouldn't block the rest
        finally:
            bar.progress((i + 1) / len(to_do), text=f"Auto-analysing {len(to_do)} runs… ({i+1}/{len(to_do)})")

    bar.empty()

    # One batch push to GitHub after all runs are done (avoids N separate commits)
    if analysed > 0:
        try:
            from cloud_storage import push_runs
            push_runs(message=f"Auto-analyse {analysed} runs")
        except Exception:
            pass

    return analysed


def _pace_str(pace_s: float) -> str:
    if not pace_s or np.isnan(pace_s):
        return "—"
    m, s = int(pace_s // 60), int(pace_s % 60)
    return f"{m}:{s:02d} /km"


def _load_history() -> pd.DataFrame | None:
    if CLASSIFIED_CSV.exists():
        try:
            df = pd.read_csv(CLASSIFIED_CSV)
            return df if len(df) > 0 else None
        except Exception:
            pass
    return None


def compute_freshness_index(df_hist: pd.DataFrame) -> pd.DataFrame | None:
    date_col = next((c for c in ["date", "start_time"] if c in df_hist.columns), None)
    if date_col is None:
        return None
    df = df_hist.copy()
    # Force numeric types so stray strings don't cause issues
    df["CML_z"] = pd.to_numeric(df["CML_z"], errors="coerce")
    df["GPV_z"] = pd.to_numeric(df["GPV_z"], errors="coerce")
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    # Drop rows with missing values OR bogus dates (anything before year 2000)
    df = df.dropna(subset=["_date", "CML_z", "GPV_z"])
    df = df[df["_date"] >= pd.Timestamp("2000-01-01")]
    df["_date"] = df["_date"].dt.date
    if len(df) < 2:
        return None
    df["cml_load"] = df["CML_z"].abs()
    df["gpv_load"] = df["GPV_z"].abs()
    date_range = pd.date_range(df["_date"].min(), df["_date"].max(), freq="D")
    daily = pd.DataFrame({"date": date_range})
    daily["_date"] = daily["date"].dt.date
    merged = daily.merge(
        df[["_date", "cml_load", "gpv_load"]].groupby("_date").mean().reset_index(),
        on="_date", how="left",
    ).fillna(0)
    for m in ["cml", "gpv"]:
        merged[f"{m}_atl"] = merged[f"{m}_load"].ewm(span=7, adjust=False).mean()
        merged[f"{m}_ctl"] = merged[f"{m}_load"].ewm(span=42, adjust=False).mean()
        merged[f"{m}_tsb"] = merged[f"{m}_ctl"] - merged[f"{m}_atl"]
    return merged


def _freshness_chart(fi: pd.DataFrame):
    # Series catalogue: (label shown in toggle, df column, color, dash, fill)
    ALL_SERIES = [
        ("Body fitness — CML CTL",    "cml_ctl", "#c0392b", "dash",  False),
        ("Body freshness — CML TSB",  "cml_tsb", "#e74c3c", "solid", True),
        ("Reg. fitness — GPV CTL",    "gpv_ctl", "#8e44ad", "dash",  False),
        ("Reg. freshness — GPV TSB",  "gpv_tsb", "#9b59b6", "solid", True),
        ("Body load — CML ATL",       "cml_atl", "#e67e22", "dot",   False),
        ("Reg. load — GPV ATL",       "gpv_atl", "#3498db", "dot",   False),
    ]
    DEFAULT_VISIBLE = [
        "Body fitness — CML CTL",
        "Body freshness — CML TSB",
        "Reg. fitness — GPV CTL",
        "Reg. freshness — GPV TSB",
    ]
    all_labels = [s[0] for s in ALL_SERIES]

    selected = st.multiselect(
        "Show series",
        options=all_labels,
        default=DEFAULT_VISIBLE,
        key="freshness_series",
        label_visibility="collapsed",
    )

    fig = go.Figure()
    for label, col, color, dash, do_fill in ALL_SERIES:
        if label not in selected:
            continue
        fill_arg = "tozeroy" if do_fill else "none"
        fill_color = "rgba(100,100,100,0.07)" if do_fill else None
        trace = go.Scatter(
            x=fi["date"],
            y=fi[col],
            name=label,
            line=dict(color=color, width=2.5 if do_fill else 1.8, dash=dash),
            fill=fill_arg,
            **({"fillcolor": fill_color} if fill_color else {}),
        )
        fig.add_trace(trace)

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(180,180,180,0.4)")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Load / Freshness (z-score units)",
        legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
        height=380,
        margin=dict(l=50, r=20, t=10, b=100),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Status row
    col_cml, col_gpv = st.columns(2)
    with col_cml:
        tsb = fi.iloc[-1]["cml_tsb"]
        if tsb > 0.1:
            st.success(f"Body (CML-TSB) = **{tsb:.2f}** — relatively **fresh**")
        elif tsb < -0.1:
            st.warning(f"Body (CML-TSB) = **{tsb:.2f}** — accumulating **physical fatigue**")
        else:
            st.info(f"Body (CML-TSB) = **{tsb:.2f}** — neutral load")
    with col_gpv:
        tsb = fi.iloc[-1]["gpv_tsb"]
        if tsb > 0.1:
            st.success(f"Regulation (GPV-TSB) = **{tsb:.2f}** — pacing **stable**")
        elif tsb < -0.1:
            st.warning(f"Regulation (GPV-TSB) = **{tsb:.2f}** — **CNS fatigue** building")
        else:
            st.info(f"Regulation (GPV-TSB) = **{tsb:.2f}** — neutral")


def _history_section(df_hist: pd.DataFrame, current_result: dict | None = None):
    """Render the full history section: freshness index + quadrant + table."""

    # ── Freshness index ───────────────────────────────────────────────────────
    st.subheader("Rolling Fatigue–Freshness Index")
    st.caption("CTL = chronic fitness (42-day EWMA)  ·  TSB = freshness (CTL − ATL)  ·  ATL = short-term load (7-day EWMA)")
    fi = compute_freshness_index(df_hist)
    if fi is not None:
        _freshness_chart(fi)
    else:
        st.info("Need at least 2 runs with dates to compute the freshness index.")

    # ── Quadrant scatter ──────────────────────────────────────────────────────
    st.subheader("All runs — fatigue quadrant")
    if "CML_z" in df_hist.columns and len(df_hist) >= 2:
        fig = px.scatter(
            df_hist, x="GPV_z", y="CML_z", color="final_label",
            hover_data=[c for c in ["run_id", "date", "session_type"] if c in df_hist.columns],
            color_discrete_map=COLOR_MAP,
        )
        if current_result:
            fig.add_scatter(
                x=[current_result["GPV_z"]], y=[current_result["CML_z"]],
                mode="markers",
                marker=dict(size=22, symbol="star", color="black",
                            line=dict(width=1.5, color="white")),
                name="This run ★",
            )
        fig.add_hline(y=0, line_dash="dash", line_color="#ccc")
        fig.add_vline(x=0, line_dash="dash", line_color="#ccc")
        all_x = df_hist["GPV_z"].abs().max() if len(df_hist) else 1
        all_y = df_hist["CML_z"].abs().max() if len(df_hist) else 1
        x_r = max(all_x, abs(current_result["GPV_z"]) if current_result else 0, 1) * 1.15
        y_r = max(all_y, abs(current_result["CML_z"]) if current_result else 0, 1) * 1.15
        fig.update_layout(
            annotations=[
                dict(x=-x_r*.75, y= y_r*.88, text="A — Body-limited",       showarrow=False, font=dict(color=TYPE_COLORS["A"], size=11)),
                dict(x= x_r*.75, y= y_r*.88, text="C — Global fatigue",     showarrow=False, font=dict(color=TYPE_COLORS["C"], size=11)),
                dict(x=-x_r*.75, y=-y_r*.88, text="D — Fresh",              showarrow=False, font=dict(color=TYPE_COLORS["D"], size=11)),
                dict(x= x_r*.75, y=-y_r*.88, text="B — Regulation-limited", showarrow=False, font=dict(color=TYPE_COLORS["B"], size=11)),
            ],
            height=420, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Run table ─────────────────────────────────────────────────────────────
    st.subheader("All runs")
    show_cols = [c for c in ["date", "run_id", "session_type", "final_label", "CML_z", "GPV_z", "neuro_tag"] if c in df_hist.columns]
    st.dataframe(df_hist[show_cols].sort_values("date", ascending=False) if "date" in df_hist.columns else df_hist[show_cols],
                 use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏃 Fatigue Detector")
    st.caption("Central vs peripheral fatigue classifier")
    st.divider()

    source = st.radio("Data source",
                      ["Garmin Connect", "Strava", "Upload TCX file"],
                      label_visibility="collapsed")

    if source == "Garmin Connect":
        st.subheader("Garmin Connect")
        email = st.text_input("Email", value=os.environ.get("GARMIN_EMAIL", ""), key="g_email")
        password = st.text_input("Password", type="password",
                                 value=os.environ.get("GARMIN_PASSWORD", ""), key="g_pass")
        days = st.slider("Look back (days)", 7, 90, 30, key="g_days")
        if st.button("Load recent runs", key="g_load"):
            with st.spinner("Connecting to Garmin…"):
                try:
                    from garmin_data import get_recent_runs
                    acts = get_recent_runs(email, password, days=days)
                    st.session_state["activities"] = acts
                    st.session_state["source"] = "garmin"
                    st.session_state["g_email_cached"] = email
                    st.session_state["g_pass_cached"] = password
                    _patch_missing_dates(acts)
                    n_new = _auto_analyse_all(acts, "garmin", g_email=email, g_password=password)
                    msg = f"Found {len(acts)} runs"
                    if n_new:
                        msg += f" · auto-analysed {n_new} new"
                    (st.success(msg) if acts else st.warning("No runs found."))
                except Exception as e:
                    st.error(f"Garmin error: {e}")

    elif source == "Strava":
        st.subheader("Strava")
        missing = [k for k in ("STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN")
                   if not os.environ.get(k)]
        if missing:
            st.warning("Missing from `.env`:\n" + "\n".join(f"- `{k}`" for k in missing))
        else:
            st.success("Strava credentials found ✓")
        days = st.slider("Look back (days)", 7, 90, 30, key="s_days")
        if st.button("Load recent runs", key="s_load"):
            with st.spinner("Fetching from Strava…"):
                try:
                    from strava_data import get_recent_runs
                    acts = get_recent_runs(days=days)
                    st.session_state["activities"] = acts
                    st.session_state["source"] = "strava"
                    _patch_missing_dates(acts)
                    n_new = _auto_analyse_all(acts, "strava")
                    msg = f"Found {len(acts)} runs"
                    if n_new:
                        msg += f" · auto-analysed {n_new} new"
                    (st.success(msg) if acts else st.warning("No runs found."))
                except Exception as e:
                    st.error(f"Strava error: {e}")

    elif source == "Upload TCX file":
        st.subheader("Upload")
        uploaded = st.file_uploader("TCX file", type=["tcx"], label_visibility="collapsed")
        if uploaded:
            st.session_state["activities"] = [{"id": uploaded.name, "name": uploaded.name,
                                               "date": "", "distance_km": 0,
                                               "source": "upload", "_file": uploaded}]
            st.session_state["source"] = "upload"

    activities: list[dict] = st.session_state.get("activities", [])
    if activities:
        st.divider()
        st.subheader("Select run")
        labels = [f"{a.get('date','')}  {a['name']}  ({a.get('distance_km',0)} km)"
                  if a.get("date") else a["name"] for a in activities]
        idx = st.selectbox("Run", range(len(activities)),
                           format_func=lambda i: labels[i], key="act_idx")
        selected = activities[idx]

        st.divider()
        st.subheader("Session type")
        session_type = st.selectbox("Session type", list(SESSION_OPTIONS.keys()),
                                    format_func=lambda k: SESSION_OPTIONS[k],
                                    key="session_type", label_visibility="collapsed")
        st.subheader("How did it feel?")
        phrase_key = st.selectbox("How did it feel?", list(NEURO_PHRASES.keys()),
                                  format_func=lambda k: NEURO_PHRASES[k],
                                  key="neuro_phrase", label_visibility="collapsed")
        neuro_tag = PHRASE_TO_TAG[phrase_key]

        if st.button("🔍 Analyse this run", type="primary", use_container_width=True):
            with st.spinner("Fetching and analysing…"):
                try:
                    src = st.session_state.get("source", "upload")
                    run_id = str(selected["id"])

                    if src == "upload" or selected.get("source") == "upload":
                        file_obj = selected["_file"]
                        with tempfile.NamedTemporaryFile(suffix=".tcx", delete=False) as tmp:
                            tmp.write(file_obj.read())
                            tmp_path = tmp.name
                        df_raw = parse_tcx(tmp_path)
                        os.unlink(tmp_path)
                    elif src == "garmin":
                        from garmin_data import download_tcx
                        tmp_path = download_tcx(st.session_state.get("g_email_cached"),
                                                st.session_state.get("g_pass_cached"),
                                                selected["id"])
                        df_raw = parse_tcx(tmp_path)
                        os.unlink(tmp_path)
                    elif src == "strava":
                        from strava_data import get_activity_dataframe
                        df_raw = get_activity_dataframe(selected["id"])

                    result, _, df_1s = run_single(df_raw, run_id=run_id,
                                                   neuro_tag=neuro_tag,
                                                   session_type=session_type,
                                                   run_date=selected.get("date"))
                    st.session_state["result"] = result
                    st.session_state["df_1s"] = df_1s
                    st.session_state["run_id"] = run_id

                except Exception as exc:
                    import traceback
                    st.error(f"Analysis failed: {exc}")
                    st.code(traceback.format_exc())


# ── Main panel ────────────────────────────────────────────────────────────────
result: dict | None = st.session_state.get("result")
df_hist = _load_history()

if result is None:
    # Landing: no run analysed yet
    st.markdown("## 🏃 Welcome to Fatigue Detector")
    st.markdown("Connect **Garmin** or **Strava** in the sidebar, pick a run, and hit **Analyse**.")
    if df_hist is not None:
        st.divider()
        _history_section(df_hist, current_result=None)
else:
    # ── Single run analysis ───────────────────────────────────────────────────
    df_1s: pd.DataFrame = st.session_state["df_1s"]
    run_id: str = st.session_state.get("run_id", "")

    code  = result.get("final_code", "M")
    color = TYPE_COLORS.get(code, TYPE_COLORS["M"])
    label = result.get("final_label", "Mixed / normal")

    # Headline
    col_label, col_cml, col_gpv, col_dist, col_dur = st.columns([3, 1, 1, 1, 1])
    with col_label:
        st.markdown(
            f"<div style='background:{color}18; border-left:5px solid {color}; "
            f"padding:14px 18px; border-radius:8px;'>"
            f"<h2 style='color:{color}; margin:0; font-size:1.5rem;'>{label}</h2>"
            f"<p style='margin:2px 0 0 0; color:#888; font-size:0.85rem;'>{run_id}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    dist_km  = (df_1s["distance_m"].iloc[-1] - df_1s["distance_m"].iloc[0]) / 1000 if "distance_m" in df_1s else 0
    dur_min  = df_1s["elapsed_s"].iloc[-1] / 60 if "elapsed_s" in df_1s else 0
    mean_pace = df_1s["pace_s_per_km"].median() if "pace_s_per_km" in df_1s else np.nan
    mean_hr   = df_1s["hr"].mean() if "hr" in df_1s else np.nan

    with col_cml:
        st.metric("CML score", f"{result['CML_z']:.2f}",
                  help="Cardio-Metabolic Load z-score. Positive = higher strain than your average.")
    with col_gpv:
        st.metric("GPV score", f"{result['GPV_z']:.2f}",
                  help="Gait/Pacing Variability z-score. Positive = more erratic than your average.")
    with col_dist:
        st.metric("Distance", f"{dist_km:.1f} km")
    with col_dur:
        st.metric("Duration", f"{dur_min:.0f} min")

    # Explanation
    with st.expander("Why this classification?", expanded=True):
        for part in result.get("reasons", "").split("; "):
            if part.strip():
                st.markdown(f"- {part.strip()}")
        if code in TYPE_EXPLANATIONS:
            st.info(TYPE_EXPLANATIONS[code])
        if code in TRAINING_TIPS:
            st.success(TRAINING_TIPS[code])

    # Run charts
    st.divider()
    tab_pace, tab_cadence, tab_terrain = st.tabs(["📈 Pace & HR", "🦵 Cadence", "⛰ Terrain"])
    elapsed_min = df_1s["elapsed_s"] / 60 if "elapsed_s" in df_1s else pd.Series(dtype=float)
    total_min   = float(elapsed_min.iloc[-1]) if len(elapsed_min) else 1.0

    with tab_pace:
        fig = go.Figure()
        pace_smooth = df_1s["pace_s_per_km"].rolling(30, center=True, min_periods=5).mean() / 60
        fig.add_trace(go.Scatter(x=elapsed_min, y=pace_smooth, name="Pace (min/km)",
                                 line=dict(color="#3498db", width=2)))
        hr_smooth = df_1s["hr"].rolling(30, center=True, min_periods=5).mean()
        fig.add_trace(go.Scatter(x=elapsed_min, y=hr_smooth, name="HR (bpm)",
                                 yaxis="y2", line=dict(color="#e74c3c", width=2)))
        fig.add_vline(x=total_min * 0.3, line_dash="dot", line_color="#bbb",
                      annotation_text="early end", annotation_position="top right")
        fig.add_vline(x=total_min * 0.7, line_dash="dot", line_color="#bbb",
                      annotation_text="late start", annotation_position="top left")
        fig.update_layout(
            xaxis_title="Time (min)",
            yaxis=dict(title="Pace (min/km)", autorange="reversed"),
            yaxis2=dict(title="HR (bpm)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.18),
            height=360, margin=dict(l=50, r=60, t=20, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg pace", _pace_str(mean_pace))
        c2.metric("Avg HR", f"{mean_hr:.0f} bpm" if not np.isnan(mean_hr) else "—")
        c3.metric("CML z-score", f"{result['CML_z']:.2f}")
        c4.metric("GPV z-score", f"{result['GPV_z']:.2f}")

    with tab_cadence:
        fig2 = go.Figure()
        cad_smooth = df_1s["cadence_spm"].rolling(30, center=True, min_periods=5).mean()
        fig2.add_trace(go.Scatter(x=elapsed_min, y=cad_smooth, name="Cadence (spm)",
                                  line=dict(color="#2ecc71", width=2)))
        fig2.add_vline(x=total_min * 0.3, line_dash="dot", line_color="#bbb")
        fig2.add_vline(x=total_min * 0.7, line_dash="dot", line_color="#bbb")
        fig2.update_layout(xaxis_title="Time (min)", yaxis_title="Cadence (steps/min)",
                           height=320, margin=dict(l=50, r=20, t=20, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    with tab_terrain:
        if "elev_m" in df_1s.columns and df_1s["elev_m"].notna().any():
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=elapsed_min, y=df_1s["elev_m"], name="Elevation (m)",
                                      fill="tozeroy", line=dict(color="#f39c12", width=1.5)))
            fig3.update_layout(xaxis_title="Time (min)", yaxis_title="Elevation (m)",
                               height=260, margin=dict(l=50, r=20, t=20, b=40))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No elevation data in this activity.")

    # ── History section below single run ──────────────────────────────────────
    if df_hist is not None:
        st.divider()
        _history_section(df_hist, current_result=result)
