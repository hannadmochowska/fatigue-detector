import numpy as np

# Internal tags you will use in the notebook:
# "body_limited", "regulation_limited", "global_fatigue", "fresh", or None.

NEURO_ADJUSTMENTS = {
    "body_limited":      {"dCML": +0.7, "dGPV": 0.0,  "note": "body-limited self-report"},
    "regulation_limited":{"dCML": 0.0,  "dGPV": +0.7, "note": "regulation/mental self-report"},
    "global_fatigue":    {"dCML": +0.7, "dGPV": +0.7, "note": "global fatigue self-report"},
    "fresh":             {"dCML": -0.7, "dGPV": -0.7, "note": "fresh / controlled self-report"},
}


def apply_neuro_adjustment(cml_z: float, gpv_z: float, neuro_tag: str | None):
    """
    Shift CML/GPV based on subjective neuro tag.
    Assumes CML_z and GPV_z are already z-scored.
    """
    if neuro_tag is None or neuro_tag not in NEURO_ADJUSTMENTS:
        return cml_z, gpv_z, None

    adj = NEURO_ADJUSTMENTS[neuro_tag]
    cml_adj = cml_z + adj["dCML"]
    gpv_adj = gpv_z + adj["dGPV"]
    return cml_adj, gpv_adj, adj["note"]

def _session_adjusted_thresholds(session_type: str | None,
                                 base_hi: float = 0.5,
                                 base_lo: float = -0.5):
    """
    Adjust high/low thresholds for CML/GPV based on session type.

    Idea:
      - Intervals tolerate higher GPV before being 'regulation-limited'.
      - Races tolerate higher CML and GPV.
      - Long runs slightly more tolerant on CML.
      - Easy runs are stricter (lower hi threshold).
    """
    if session_type is None:
        return base_hi, base_lo, base_hi, base_lo  # hi_cml, lo_cml, hi_gpv, lo_gpv

    st = session_type.lower()

    hi_cml = base_hi
    lo_cml = base_lo
    hi_gpv = base_hi
    lo_gpv = base_lo

    if st == "interval":
        hi_gpv = base_hi + 0.4  # allow more pacing variability
    elif st == "race":
        hi_cml = base_hi + 0.4
        hi_gpv = base_hi + 0.3
    elif st == "long":
        hi_cml = base_hi + 0.2
    elif st == "easy":
        hi_cml = base_hi - 0.1
        hi_gpv = base_hi - 0.1

    return hi_cml, lo_cml, hi_gpv, lo_gpv


def classify_from_scores(cml_z: float,
                         gpv_z: float,
                         session_type: str | None = None,
                         hi: float = 0.5,
                         lo: float = -0.5):
    """
    Purely objective classification from CML and GPV (no neuro input),
    with thresholds adjusted for session type.
    """
    hi_cml, lo_cml, hi_gpv, lo_gpv = _session_adjusted_thresholds(session_type, hi, lo)

    cml_high = cml_z > hi_cml
    cml_low  = cml_z < lo_cml
    gpv_high = gpv_z > hi_gpv
    gpv_low  = gpv_z < lo_gpv

    if cml_high and gpv_low:
        label = "Type A — Body-limited"
        short = "A"
    elif cml_low and gpv_high:
        label = "Type B — Regulation-limited"
        short = "B"
    elif cml_high and gpv_high:
        label = "Type C — Global fatigue"
        short = "C"
    elif cml_low and gpv_low:
        label = "Type D — Fresh / controlled"
        short = "D"
    else:
        label = "Mixed / normal"
        short = "M"

    return label, short



def classify_run(cml_z: float,
                 gpv_z: float,
                 neuro_tag: str | None = None,
                 session_type: str | None = None,
                 hi: float = 0.5,
                 lo: float = -0.5) -> dict:
    """
    Full classifier:
    - takes z-scored CML & GPV
    - adjusts thresholds based on session type
    - applies quadrant logic
    - optionally shifts scores based on neuro_tag
    """
    base_label, base_code = classify_from_scores(cml_z, gpv_z, session_type, hi, lo)

    cml_adj, gpv_adj, note = apply_neuro_adjustment(cml_z, gpv_z, neuro_tag)
    final_label, final_code = classify_from_scores(cml_adj, gpv_adj, session_type, hi, lo)

    reasons = [f"base → {base_label} (CML_z={cml_z:.2f}, GPV_z={gpv_z:.2f})"]
    if neuro_tag is not None:
        reasons.append(
            f"neuro_tag='{neuro_tag}' ({note}); "
            f"after adj → {final_label} (CML_adj={cml_adj:.2f}, GPV_adj={gpv_adj:.2f})"
        )

    return {
        "base_label": base_label,
        "base_code": base_code,
        "final_label": final_label,
        "final_code": final_code,
        "CML_z": cml_z,
        "GPV_z": gpv_z,
        "CML_adj": cml_adj,
        "GPV_adj": gpv_adj,
        "neuro_tag": neuro_tag,
        "session_type": session_type,
        "reasons": "; ".join(reasons),
    }

