#!/usr/bin/env python3
"""
Pseudo-online CASCADE debug harness (Stage 1 -> Stage 2).

What it does
------------
- Loads a recorded FIF (from lsl_record_to_fif.py) with annotations.
- Loads TWO saved joblib payloads:
    Stage 1: MI vs REST  (binary; proba[:,1] = P(MI))
    Stage 2: FISTS vs FEET (binary; proba[:,1] = P(FEET) where 0=fists, 1=feet)
- Applies the correct preprocessing per model config (bandpass, optional resample).
- Slides windows in time order and computes:
    - P(MI) for every window
    - gated P(FEET) over time (NaN when gate is closed)

- Plots:
    - P(MI) over time
    - gated P(FEET) over time (NaN when gate is closed)
    - marker lines for T0/T1/T2 and block boundaries

Usage
-----
python Pipline/MIClassification/debug_pseudo_online_cascade.py \
  --fif Pipline/Training_Data_Acquisition/recordings/Tryout_01_MI_TRAINING_20260213_135551_raw.fif \
  --stage1 Pipline/MIClassification/MIRest/models/Stage1_MI_vs_REST_CSP_LDA_all.joblib \
  --stage2 Pipline/MIClassification/MITypes/models/Stage2_FISTS_vs_FEET_CSP_LDA_all.joblib \
  --mi-thr 0.8 --dwell 1.0 --refractory 2.0

Notes
-----
- For theater safety (avoid false triggers), you typically increase --mi-thr and/or --dwell.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import mne
import joblib
import matplotlib.pyplot as plt


# -----------------------
# Channel group definitions (must match training scripts)
# -----------------------

DISTRIBUTED_16_NAMES = [
    "Fp1", "Fp2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T7", "T8",
    "P7", "P8",
]

MOTOR_ROI_NAMES = [
    "FC3", "FCz", "FC4",
    "C3", "Cz", "C4",
    "CP3", "CPz", "CP4",
]

DISTRIBUTED_16_INDEX_FALLBACK = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
MOTOR_ROI_INDEX_FALLBACK = [6, 7, 8, 14, 15, 16, 22, 23, 24]


def resolve_picks(raw: mne.io.BaseRaw, channel_group: str) -> Tuple[np.ndarray, str]:
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude="bads")

    if channel_group == "all":
        return eeg_picks, f"all ({len(eeg_picks)} EEG channels)"

    def name_based(names: List[str]) -> Optional[np.ndarray]:
        want = [n for n in names if n in raw.ch_names]
        if not want:
            return None
        return mne.pick_channels(raw.ch_names, include=want)

    if channel_group == "distributed16":
        p = name_based(DISTRIBUTED_16_NAMES)
        if p is not None and len(p) >= 8:
            return p, f"distributed16 (name-based, n={len(p)})"
        idx = [i for i in DISTRIBUTED_16_INDEX_FALLBACK if i < len(eeg_picks)]
        idx = idx[: min(16, len(idx))]
        return eeg_picks[idx], f"distributed16 (index-fallback, n={len(idx)})"

    if channel_group == "motor_roi":
        p = name_based(MOTOR_ROI_NAMES)
        if p is not None and len(p) >= 3:
            return p, f"motor_roi (name-based, n={len(p)})"
        idx = [i for i in MOTOR_ROI_INDEX_FALLBACK if i < len(eeg_picks)]
        idx = idx[: min(len(MOTOR_ROI_INDEX_FALLBACK), len(eeg_picks))]
        return eeg_picks[idx], f"motor_roi (index-fallback, n={len(idx)})"

    raise ValueError(f"Unknown channel_group='{channel_group}'.")


# -----------------------
# Annotation helpers (for plotting + basic sanity stats)
# -----------------------

def get_markers(raw: mne.io.BaseRaw) -> List[Tuple[float, str]]:
    ann = raw.annotations
    pairs = sorted([(float(o), str(d)) for o, d in zip(ann.onset, ann.description)], key=lambda x: x[0])
    keep = []
    for t, d in pairs:
        if d in ("T0", "T1", "T2") or (d.startswith("B") and (d.endswith("_START") or d.endswith("_END"))):
            keep.append((t, d))
    return keep


def step_state_at_times(markers: List[Tuple[float, str]], t: np.ndarray) -> np.ndarray:
    states = [(ts, d) for ts, d in markers if d in ("T0", "T1", "T2")]
    out = np.array([""] * len(t), dtype=object)
    if not states:
        return out
    cur = ""
    j = 0
    for i, ti in enumerate(t):
        while j < len(states) and states[j][0] <= ti:
            cur = states[j][1]
            j += 1
        out[i] = cur
    return out


# -----------------------
# Window extraction
# -----------------------

def build_window_starts(tmin: float, tmax: float, win_len: float, step: float) -> np.ndarray:
    if tmax - tmin < win_len:
        return np.array([], dtype=float)
    return np.arange(tmin, tmax - win_len + 1e-9, step, dtype=float)


def extract_X(raw: mne.io.BaseRaw, starts: np.ndarray, win_len: float, picks: np.ndarray) -> np.ndarray:
    sfreq = raw.info["sfreq"]
    n_times = int(round(win_len * sfreq))
    X = np.zeros((len(starts), len(picks), n_times), dtype=np.float32)

    for i, s in enumerate(starts):
        a = int(round(s * sfreq))
        b = a + n_times
        seg = raw.get_data(picks=picks, start=a, stop=b)
        if seg.shape[1] != n_times:
            pad = n_times - seg.shape[1]
            seg = np.pad(seg, ((0, 0), (0, max(0, pad))), mode="constant")
            seg = seg[:, :n_times]
        X[i] = seg.astype(np.float32)
    return X


def predict_proba_batched(model, X: np.ndarray, batch: int) -> np.ndarray:
    out = np.zeros((X.shape[0],), dtype=float)
    for i in range(0, X.shape[0], batch):
        pb = model.predict_proba(X[i:i + batch])
        out[i:i + batch] = pb[:, 1]
    return out


# -----------------------
# Gate logic (threshold + dwell + refractory)
# -----------------------

def apply_gate(
    t_mid: np.ndarray,
    p_mi: np.ndarray,
    mi_thr: float,
    dwell_s: float,
    refractory_s: float,
) -> np.ndarray:
    """
    Returns boolean array gate_open per window midpoint.
    Gate opens when p_mi has been >= mi_thr for dwell_s continuously.
    After opening, a refractory period prevents re-opening events (but gate can stay open).
    """
    gate = np.zeros_like(p_mi, dtype=bool)

    if len(t_mid) < 2:
        return gate

    dt = float(np.median(np.diff(t_mid)))
    dwell_n = max(1, int(np.ceil(dwell_s / dt)))
    refr_n = int(np.ceil(refractory_s / dt))

    above = p_mi >= mi_thr
    consec = 0
    refr_count = 0
    opened_once = False

    for i in range(len(p_mi)):
        if refr_count > 0:
            refr_count -= 1

        if above[i]:
            consec += 1
        else:
            consec = 0
            opened_once = False  # require new dwell stretch to re-open

        # Open when dwell reached and not in refractory and not already opened in this stretch
        if consec >= dwell_n and refr_count == 0 and not opened_once:
            opened_once = True
            refr_count = refr_n

        # Gate stays open whenever we are currently above threshold (conservative option)
        # Alternative would be hysteresis. For now: open iff above threshold.
        gate[i] = above[i]

    return gate


# -----------------------
# Preprocessing per model
# -----------------------

def preprocess_for_model(raw_in: mne.io.BaseRaw, cfg: Dict, picks: np.ndarray) -> mne.io.BaseRaw:
    """
    Apply per-model preprocessing on a copy of raw:
      - optional notch
      - bandpass
      - optional resample
    """
    raw = raw_in.copy()

    notch = cfg.get("notch", None)
    freq_band = cfg.get("freq_band", (8.0, 30.0))
    resample_sfreq = cfg.get("resample_sfreq", None)

    if notch is not None:
        raw.notch_filter(notch, picks=picks, verbose=False)

    raw.filter(freq_band[0], freq_band[1], picks=picks, verbose=False)

    if resample_sfreq is not None:
        raw.resample(resample_sfreq, npad="auto")

    return raw


# -----------------------
# Plot
# -----------------------

def plot_cascade(
    t_mid: np.ndarray,
    p_mi: np.ndarray,
    p_feet_gated: np.ndarray,
    gate: np.ndarray,
    markers: List[Tuple[float, str]],
    title: str,
):
    fig, ax = plt.subplots()
    ax.plot(t_mid, p_mi, label="Stage 1: P(MI)")
    ax.plot(t_mid, p_feet_gated, label="Stage 2: P(FEET) (gated)")

    # show gate as a faint band at bottom
    gate_y = np.where(gate, 0.02, np.nan)
    ax.plot(t_mid, gate_y, label="Gate open", linewidth=3)

    # marker lines
    for ts, d in markers:
        if d in ("T0", "T1", "T2"):
            ax.axvline(ts, linestyle="--", linewidth=1)
        elif d.endswith("_START") or d.endswith("_END"):
            ax.axvline(ts, linestyle=":", linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.show()


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fif", required=True, type=str)
    ap.add_argument("--stage1", required=True, type=str, help="Stage1 joblib (MI vs REST)")
    ap.add_argument("--stage2", required=True, type=str, help="Stage2 joblib (FISTS vs FEET)")
    ap.add_argument("--mi-thr", type=float, default=0.8, help="Threshold on P(MI) to open gate")
    ap.add_argument("--dwell", type=float, default=1.0, help="Seconds above threshold required (for debugging stats)")
    ap.add_argument("--refractory", type=float, default=2.0, help="Seconds refractory after opening event")
    ap.add_argument("--step", type=float, default=None, help="Window step seconds (default: stage1 win_step or 0.25)")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    fif_path = Path(args.fif)

    p1 = joblib.load(Path(args.stage1))
    p2 = joblib.load(Path(args.stage2))

    m1 = p1["model"]
    m2 = p2["model"]

    cfg1: Dict = p1.get("config", {})
    cfg2: Dict = p2.get("config", {})

    # Load raw once
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # Resolve picks separately for each model (they may differ!)
    grp1 = cfg1.get("channel_group", "all")
    grp2 = cfg2.get("channel_group", "all")

    picks1, desc1 = resolve_picks(raw, grp1)
    picks2, desc2 = resolve_picks(raw, grp2)

    print(f"[INFO] Stage1 channel group: {grp1} -> {desc1}")
    print(f"[INFO] Stage2 channel group: {grp2} -> {desc2}")

    # Preprocess per model on copies
    raw1 = preprocess_for_model(raw, cfg1, picks1)
    raw2 = preprocess_for_model(raw, cfg2, picks2)

    # Windowing: use Stage 1 win_len as the master timeline
    win_len1 = float(cfg1.get("win_len", 2.0))
    win_step1 = float(cfg1.get("win_step", 0.25))
    step = args.step if args.step is not None else win_step1

    # If Stage 2 has different win_len, we will extract its windows aligned to the same starts.
    win_len2 = float(cfg2.get("win_len", win_len1))

    rec_end = min(raw1.times[-1], raw2.times[-1])
    tmin = max(0.0, args.tmin)
    tmax = rec_end if args.tmax is None else min(rec_end, args.tmax)

    starts = build_window_starts(tmin, tmax, win_len1, step)
    if len(starts) == 0:
        raise RuntimeError("No windows fit into selected range.")

    t_mid = starts + win_len1 / 2.0

    print(f"[INFO] Timeline windows: WIN_LEN1={win_len1}s, STEP={step}s, n={len(starts)}")

    # Stage 1 inference
    X1 = extract_X(raw1, starts, win_len=win_len1, picks=picks1)
    p_mi = predict_proba_batched(m1, X1, batch=args.batch)

    # Gate
    gate = apply_gate(t_mid, p_mi, mi_thr=args.mi_thr, dwell_s=args.dwell, refractory_s=args.refractory)

    # Stage 2 inference (only compute where gate is open)
    p_feet = np.full_like(p_mi, np.nan, dtype=float)
    idx = np.where(gate)[0]
    if len(idx) > 0:
        # For stage2, we extract windows aligned to the same start times.
        starts2 = starts[idx]
        X2 = extract_X(raw2, starts2, win_len=win_len2, picks=picks2)
        p2_out = predict_proba_batched(m2, X2, batch=args.batch)
        p_feet[idx] = p2_out

    markers = get_markers(raw)

    # Basic debug summary based on annotation states (optional but useful)
    states_mid = step_state_at_times(markers, t_mid)
    for key in ("T0", "T1", "T2"):
        mask = states_mid == key
        if mask.any():
            print(f"[DEBUG] mean P(MI) during {key}: {p_mi[mask].mean():.3f} (n={mask.sum()})")
    # For Stage 2, meaningful only in T1/T2
    m1_mask = states_mid == "T1"
    m2_mask = states_mid == "T2"
    if np.any(m1_mask & gate):
        print(f"[DEBUG] mean P(FEET) during T1 (fists) when gated: {np.nanmean(p_feet[m1_mask]):.3f}")
    if np.any(m2_mask & gate):
        print(f"[DEBUG] mean P(FEET) during T2 (feet) when gated: {np.nanmean(p_feet[m2_mask]):.3f}")

    title = (
        f"Cascade on {fif_path.name}\n"
        f"Stage1={Path(args.stage1).name} | Stage2={Path(args.stage2).name} | "
        f"mi_thr={args.mi_thr}, dwell={args.dwell}s, refr={args.refractory}s"
    )
    plot_cascade(t_mid, p_mi, p_feet, gate, markers, title)


if __name__ == "__main__":
    main()