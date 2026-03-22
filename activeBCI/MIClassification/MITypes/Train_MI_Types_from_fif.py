"""
Stage-2 training: MI TYPES (FISTS vs FEET) using CSP + LDA from your own LSL-recorded data.

Input produced by lsl_record_to_fif.py:
  - <SUBJECT>_<SESSION>_<timestamp>_raw.fif   (MNE Raw with annotations)

Annotation convention (from mi_cues_markers.py):
  - T1 = FISTS MI
  - T2 = FEET  MI
  - T0 = REST (ignored for stage 2)
  - optional: B1_START/B1_END etc.

Key difference vs PhysioNet script:
- PhysioNet uses trial onsets + fixed [TMIN,TMAX] epochs.
- Here we use state-change markers => continuous segments labeled T1 or T2.
- We extract sliding windows from inside each segment (excluding transition buffer).

Outputs:
  - joblib model payload to Pipline/MIClassification/MITypes/models/
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import mne
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import confusion_matrix
import joblib


# -----------------------
# CONFIG
# -----------------------

RECORDINGS_DIR = Path("Pipline/Training_Data_Acquisition/recordings")

# Choose input FIF:
RAW_FIF_PATH: Optional[Path] = None  # set explicitly OR use AUTO_PICK_NEWEST
AUTO_PICK_NEWEST = True
SUBJECT_FILTER = "Tryout_01"
SESSIONTAG_FILTER = "MI_TRAINING"

# Preprocessing (similar spirit to your PhysioNet stage-2 script)
FREQ_BAND = (8.0, 20.0)   # you used (8-20) in physionet stage2; keep unless you decide otherwise
NOTCH = None              # set 50.0 or 60.0 if needed
RESAMPLE_SFREQ = None     # e.g. 250.0 once you know device rate; None keeps native

# -----------------------
# Channel selection
# -----------------------
# "all"           : use all EEG channels available
# "distributed16" : use 16 channels distributed over the scalp (robust / less overfitting)
# "motor_roi"     : use motor-area channels (ideal for MI: around C3/Cz/C4 + neighbors)
CHANNEL_GROUP = "all"

# If your channels are properly named (e.g., C3, Cz, C4, ...), we can use name-based groups.
# If channels are still generic (EEG01..EEG32), the script will fall back to index-based picks.
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

# Index-based fallbacks (used if the channel names above are not present)
DISTRIBUTED_16_INDEX_FALLBACK = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
MOTOR_ROI_INDEX_FALLBACK = [6, 7, 8, 14, 15, 16, 22, 23, 24]

# Labels
FISTS_MARKER = "T1"
FEET_MARKER = "T2"
USE_BLOCKS: Optional[Tuple[str, ...]] = None  # e.g. ("B1", "B2") or None for all

# Windowing inside state segments
WIN_LEN = 2.0             # seconds
WIN_STEP = 0.25           # seconds
TRANSITION_GUARD = 1.0    # seconds ignored after each state change

# CSP/LDA
CSP_COMPONENTS = 4
CSP_REG = "ledoit_wolf"

# Validation
# IMPORTANT: avoid leakage from overlapping windows by grouping windows by their parent segment.
CV_SPLITS = 5

# Output
MODEL_DIR = Path("Pipline/MIClassification/MITypes/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = f"Stage2_FISTS_vs_FEET_CSP_LDA_{CHANNEL_GROUP}.joblib"


# -----------------------
# Data structures
# -----------------------

@dataclass
class TypeSegment:
    onset: float         # seconds
    offset: float        # seconds
    y: int               # 0=fists, 1=feet
    seg_id: int          # unique id for grouping (CV)
    block: Optional[str] = None  # "B1"/"B2"/"B3" if available


# -----------------------
# Helpers
# -----------------------

def newest_fif(recordings_dir: Path, subject: str, sessiontag: str) -> Path:
    cands = sorted(recordings_dir.glob(f"{subject}_{sessiontag}_*_raw.fif"))
    if not cands:
        raise FileNotFoundError(f"No FIF files found matching {subject}_{sessiontag}_*_raw.fif in {recordings_dir}")
    return cands[-1]


def resolve_picks(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, str]:
    """Resolve EEG channel picks based on CHANNEL_GROUP."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude="bads")

    if CHANNEL_GROUP == "all":
        return eeg_picks, f"all ({len(eeg_picks)} EEG channels)"

    def name_based(names: List[str]) -> Optional[np.ndarray]:
        want = [n for n in names if n in raw.ch_names]
        if not want:
            return None
        return mne.pick_channels(raw.ch_names, include=want)

    if CHANNEL_GROUP == "distributed16":
        p = name_based(DISTRIBUTED_16_NAMES)
        if p is not None and len(p) >= 8:
            return p, f"distributed16 (name-based, n={len(p)})"
        idx = [i for i in DISTRIBUTED_16_INDEX_FALLBACK if i < len(eeg_picks)]
        idx = idx[: min(16, len(idx))]
        return eeg_picks[idx], f"distributed16 (index-fallback, n={len(idx)})"

    if CHANNEL_GROUP == "motor_roi":
        p = name_based(MOTOR_ROI_NAMES)
        if p is not None and len(p) >= 3:
            return p, f"motor_roi (name-based, n={len(p)})"
        idx = [i for i in MOTOR_ROI_INDEX_FALLBACK if i < len(eeg_picks)]
        idx = idx[: min(len(MOTOR_ROI_INDEX_FALLBACK), len(eeg_picks))]
        return eeg_picks[idx], f"motor_roi (index-fallback, n={len(idx)})"

    raise ValueError(
        f"Unknown CHANNEL_GROUP='{CHANNEL_GROUP}'. Use 'all', 'distributed16', or 'motor_roi'."
    )


def extract_blocks(raw: mne.io.BaseRaw) -> List[Tuple[str, float, float]]:
    """Return list of (block_id, start_s, end_s) if B*_START/B*_END exist."""
    ann = raw.annotations
    starts = {}
    ends = {}
    for onset, desc in zip(ann.onset, ann.description):
        desc = str(desc)
        if desc.startswith("B") and desc.endswith("_START"):
            starts[desc.replace("_START", "")] = float(onset)
        if desc.startswith("B") and desc.endswith("_END"):
            ends[desc.replace("_END", "")] = float(onset)

    out = []
    for b, s in starts.items():
        e = ends.get(b, None)
        if e is not None and e > s:
            out.append((b, s, e))
    return sorted(out, key=lambda x: x[1])


def block_of_time(blocks: List[Tuple[str, float, float]], t: float) -> Optional[str]:
    for b, s, e in blocks:
        if s <= t <= e:
            return b
    return None


def build_type_segments_from_state_changes(raw: mne.io.BaseRaw) -> List[TypeSegment]:
    """
    Convert state-change markers into continuous segments labeled as fists/feet.

    Assumption:
      - A marker T1 or T2 means "from now until next state marker, the MI type is T1/T2".
      - T0 (rest) is ignored.
    """
    ann = raw.annotations
    pairs = sorted([(float(o), str(d)) for o, d in zip(ann.onset, ann.description)], key=lambda x: x[0])

    state_keys = (FISTS_MARKER, FEET_MARKER, "T0")
    states = [(t, d) for t, d in pairs if d in state_keys]
    if len(states) < 2:
        raise RuntimeError("Not enough state markers (T0/T1/T2) found in annotations to build segments.")

    blocks = extract_blocks(raw)

    segs: List[TypeSegment] = []
    seg_id = 0
    for (t0, s0), (t1, _s1) in zip(states[:-1], states[1:]):
        if t1 <= t0:
            continue
        if s0 not in (FISTS_MARKER, FEET_MARKER):
            continue  # skip rest segments for stage2
        y = 0 if s0 == FISTS_MARKER else 1
        segs.append(TypeSegment(onset=t0, offset=t1, y=y, seg_id=seg_id, block=block_of_time(blocks, t0)))
        seg_id += 1

    if not segs:
        raise RuntimeError("No T1/T2 segments found. Did you record MI types (T1/T2) in the cue session?")

    return segs


def windows_from_segments(
    segs: List[TypeSegment],
    win_len: float,
    win_step: float,
    guard: float,
    use_blocks: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray]:
    """
    Return:
      windows: list of (start_s, end_s)
      y:       0=fists, 1=feet per window
      groups:  segment id per window (for GroupKFold)
    """
    windows: List[Tuple[float, float]] = []
    ys: List[int] = []
    groups: List[int] = []

    for seg in segs:
        if use_blocks is not None:
            if seg.block is None or seg.block not in use_blocks:
                continue

        start = seg.onset + guard
        end = seg.offset
        if end - start < win_len:
            continue

        t = start
        while t + win_len <= end:
            windows.append((t, t + win_len))
            ys.append(seg.y)
            groups.append(seg.seg_id)
            t += win_step

    if not windows:
        raise RuntimeError(
            "No windows extracted. Check TRANSITION_GUARD/WIN_LEN or verify that T1/T2 segments are long enough."
        )

    return windows, np.asarray(ys, dtype=int), np.asarray(groups, dtype=int)


def build_epochs_from_windows(raw: mne.io.BaseRaw, windows: List[Tuple[float, float]], win_len: float, picks: np.ndarray) -> mne.Epochs:
    """Create epochs by placing fake events at each window start."""
    sfreq = raw.info["sfreq"]
    onsets_samples = np.array([int(round(t0 * sfreq)) for (t0, _t1) in windows], dtype=int)

    events = np.column_stack(
        [onsets_samples, np.zeros(len(onsets_samples), dtype=int), np.ones(len(onsets_samples), dtype=int)]
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"win": 1},
        tmin=0.0,
        tmax=win_len,
        baseline=None,
        preload=True,
        picks=picks,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs


def summarize_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # rows=true [fists, feet], cols=pred [fists, feet]
    acc = (cm[0, 0] + cm[1, 1]) / max(1, cm.sum())
    acc_fists = cm[0, 0] / max(1, cm[0, :].sum())
    acc_feet = cm[1, 1] / max(1, cm[1, :].sum())
    print("Confusion matrix (rows=true fists/feet, cols=pred fists/feet):")
    print(cm)
    print(f"Window-acc={acc:.3f} | per-class acc: fists={acc_fists:.3f} feet={acc_feet:.3f}")


# -----------------------
# Main
# -----------------------

def main():
    # Pick input
    if RAW_FIF_PATH is not None:
        fif_path = Path(RAW_FIF_PATH)
    elif AUTO_PICK_NEWEST:
        fif_path = newest_fif(RECORDINGS_DIR, SUBJECT_FILTER, SESSIONTAG_FILTER)
    else:
        raise ValueError("Set RAW_FIF_PATH or enable AUTO_PICK_NEWEST.")

    print(f"Loading: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    picks, picks_desc = resolve_picks(raw)
    print(f"Channel group: {CHANNEL_GROUP} -> {picks_desc}")

    # Preprocess
    if NOTCH is not None:
        raw.notch_filter(NOTCH, picks=picks, verbose=False)
    raw.filter(FREQ_BAND[0], FREQ_BAND[1], picks=picks, verbose=False)
    if RESAMPLE_SFREQ is not None:
        raw.resample(RESAMPLE_SFREQ, npad="auto")

    # Segments -> windows
    segs = build_type_segments_from_state_changes(raw)
    blocks_present = sorted({s.block for s in segs if s.block is not None})
    print(f"Found {len(segs)} T1/T2 segments. Blocks present: {blocks_present if blocks_present else 'None'}")

    windows, y, groups = windows_from_segments(
        segs,
        win_len=WIN_LEN,
        win_step=WIN_STEP,
        guard=TRANSITION_GUARD,
        use_blocks=USE_BLOCKS,
    )
    print(f"Extracted windows: {len(windows)} | fists={int((y==0).sum())} feet={int((y==1).sum())}")

    epochs = build_epochs_from_windows(raw, windows, win_len=WIN_LEN, picks=picks)
    X = epochs.get_data()  # picks already applied in Epochs

    # Model
    clf = Pipeline(
        [
            ("csp", CSP(n_components=CSP_COMPONENTS, reg=CSP_REG, log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )

    # CV sanity check (GroupKFold avoids leakage from overlapping windows in same segment)
    n_groups = len(np.unique(groups))
    n_splits = min(CV_SPLITS, n_groups)
    if n_splits < 2:
        print("Warning: not enough segments for GroupKFold CV. Skipping CV.")
    else:
        cv = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(clf, X, y, groups=groups, cv=cv)
        print(f"GroupKFold CV acc ({n_splits}-fold, grouped by MI segment): {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit final model on all windows
    clf.fit(X, y)

    # Quick training-set summary (just for sanity)
    y_pred = clf.predict(X)
    summarize_confusion(y, y_pred)

    # Save payload
    out_path = MODEL_DIR / MODEL_NAME
    payload = {
        "model": clf,
        "config": {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "fif_path": str(fif_path),
            "freq_band": FREQ_BAND,
            "notch": NOTCH,
            "resample_sfreq": RESAMPLE_SFREQ,
            "channel_group": CHANNEL_GROUP,
            "channel_picks_desc": picks_desc,
            "win_len": WIN_LEN,
            "win_step": WIN_STEP,
            "transition_guard": TRANSITION_GUARD,
            "use_blocks": USE_BLOCKS,
            "labels": {"fists": FISTS_MARKER, "feet": FEET_MARKER},
            "csp_components": CSP_COMPONENTS,
            "csp_reg": CSP_REG,
            "cv": {"type": "GroupKFold", "grouping": "MI segment id"},
        },
    }
    joblib.dump(payload, out_path)
    print(f"Saved model: {out_path}")


if __name__ == "__main__":
    main()