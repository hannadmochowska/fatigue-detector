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


def classify_from_scores(cml_z: float, gpv_z: float, hi: float = 0.5, lo: float = -0.5):
    """
    Purely objective classification from CML and GPV (no neuro input).
    Uses hi/lo thresholds to decide quadrant.
    """
    # high / low flags
    cml_high = cml_z > hi
    cml_low  = cml_z < lo
    gpv_high = gpv_z > hi
    gpv_low  = gpv_z < lo

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


def classify_run(cml_z: float, gpv_z: float, neuro_tag: str | None = None,
                 hi: float = 0.5, lo: float = -0.5) -> dict:
    """
    Full classifier:
    - takes z-scored CML & GPV
    - optionally applies neuro adjustment
    - returns final label + intermediate info
    """

    # objective-only classification first
    base_label, base_code = classify_from_scores(cml_z, gpv_z, hi=hi, lo=lo)

    # apply neuro layer
    cml_adj, gpv_adj, neuro_note = apply_neuro_adjustment(cml_z, gpv_z, neuro_tag)
    final_label, final_code = classify_from_scores(cml_adj, gpv_adj, hi=hi, lo=lo)

    reasons = []
    reasons.append(f"objective CML_z={cml_z:.2f}, GPV_z={gpv_z:.2f} → {base_label}")
    if neuro_note:
        reasons.append(f"neuro input: {neuro_note}")
        reasons.append(f"after neuro adj → CML_z={cml_adj:.2f}, GPV_z={gpv_adj:.2f} → {final_label}")

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
        "reasons": "; ".join(reasons) if reasons else base_label,
    }
