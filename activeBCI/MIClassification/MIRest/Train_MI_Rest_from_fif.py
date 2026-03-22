"""
Train Stage-1 MI-vs-REST classifier (CSP + LDA) from your own LSL-recorded data.

Input produced by lsl_record_to_fif.py:
  - <SUBJECT>_<SESSION>_<timestamp>_raw.fif   (MNE Raw with annotations)
  - <SUBJECT>_<SESSION>_<timestamp>_events.csv (optional, debug)
  - <SUBJECT>_<SESSION>_<timestamp>_meta.json

Important: Annotations in FIF must contain state markers:
  - T0 = REST
  - T1 = FISTS MI
  - T2 = FEET MI
Optionally:
  - B1_START/B1_END, B2_START/B2_END, B3_START/B3_END for segmentation.

Training idea:
- Convert state-change annotations into labeled time intervals.
- Extract many short windows (e.g., 2s) from inside those intervals
  (excluding transition buffers).
- Label windows as REST (T0) vs MI (T1/T2 pooled).
- Fit CSP on those windows, then LDA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib


# -----------------------
# CONFIG
# -----------------------

# Where your recorder writes output:
RECORDINGS_DIR = Path("activeBCI/Training_Data_Acquisition/recordings")

# Pick the FIF you want to train on:
# Option A: set explicitly
RAW_FIF_PATH = None  # e.g. RECORDINGS_DIR / "Tryout_01_MI_TRAINING_20260213_180000_raw.fif"

# Option B: auto-pick newest matching file
AUTO_PICK_NEWEST = True
SUBJECT_FILTER = "Tryout_01"      # used only for AUTO_PICK_NEWEST
SESSIONTAG_FILTER = "MI_TRAINING" # used only for AUTO_PICK_NEWEST

# Preprocessing
FREQ_BAND = (8.0, 30.0)   # MI band
NOTCH = None              # e.g. 50.0 or 60.0 if needed; None = no notch
RESAMPLE_SFREQ = None     # e.g. 250.0 to standardize and speed up; None keeps native

# Channel selection
# "all"           : use all EEG channels available
# "distributed16" : use 16 channels distributed over the scalp (robust / less overfitting)
# "motor_roi"     : use motor-area channels (ideal for MI: around C3/Cz/C4 + neighbors)
CHANNEL_GROUP = "all"

# If your channels are properly named (e.g., C3, Cz, C4, ...), you can define explicit groups.
# If channels are still generic (EEG01..EEG32), the script will fall back to index-based picks.
DISTRIBUTED_16_NAMES = [
    # Broad scalp coverage (edit later to match your exact 32-ch layout)
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
    # Motor ROI (edit later to match your cap): hand/foot MI strongest here
    "FC3", "FCz", "FC4",
    "C3", "Cz", "C4",
    "CP3", "CPz", "CP4",
]

# Index-based fallbacks (used if the channel names above are not present)
# distributed16: take a spread of indices across the available EEG channels
DISTRIBUTED_16_INDEX_FALLBACK = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
# motor_roi: prefer mid channels if names are unknown (tune once you know the device order)
MOTOR_ROI_INDEX_FALLBACK = [6, 7, 8, 14, 15, 16, 22, 23, 24]

# Window extraction from state segments
WIN_LEN = 2.0            # seconds (this controls stage latency)
WIN_STEP = 0.25          # seconds (how densely windows are sampled)
TRANSITION_GUARD = 1.0   # seconds ignored after each state change (cue onset transient)

# Labels
REST_MARKER = "T0"
MI_MARKERS = ("T1", "T2")  # pooled for Stage-1

# Optional: train only on some blocks (None = use all)
# Example: TRAIN_BLOCKS = ("B1", "B2")  or ("B1",) etc.
TRAIN_BLOCKS: Optional[Tuple[str, ...]] = None

# Model
CSP_COMPONENTS = 4
CSP_REG = "ledoit_wolf"   # regularized covariance helps robustness
CV_FOLDS = 5

# Output
MODEL_DIR = Path("activeBCI/MIClassification/MIRest/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = f"Stage1_MI_vs_REST_CSP_LDA_{CHANNEL_GROUP}.joblib"


# -----------------------
# Helpers
# -----------------------

def resolve_picks(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, str]:
    """Resolve channel picks based on CHANNEL_GROUP.

    Returns:
      picks: numpy array of channel indices to use
      desc:  human-readable description for logging

    Notes:
    - If channels are properly named, we use name-based groups.
    - If not, we fall back to index-based picks on the available EEG channels.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude="bads")
    eeg_names = [raw.ch_names[i] for i in eeg_picks]

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
        # fallback: spread indices over EEG picks
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

    raise ValueError(f"Unknown CHANNEL_GROUP='{CHANNEL_GROUP}'. Use 'all', 'distributed16', or 'motor_roi'.")


@dataclass
class StateSegment:
    onset: float      # seconds (raw time)
    offset: float     # seconds (raw time)
    label: str        # "rest" or "mi"
    block: Optional[str] = None  # e.g. "B1", "B2", "B3"


def newest_fif(recordings_dir: Path, subject: str, sessiontag: str) -> Path:
    cands = sorted(recordings_dir.glob(f"{subject}_{sessiontag}_*_raw.fif"))
    if not cands:
        raise FileNotFoundError(f"No FIF files found matching {subject}_{sessiontag}_*_raw.fif in {recordings_dir}")
    return cands[-1]


def apply_channel_rename(raw: mne.io.BaseRaw, rename_map: Optional[Dict[str, str]]) -> None:
    if rename_map:
        present = {k: v for k, v in rename_map.items() if k in raw.ch_names}
        if present:
            raw.rename_channels(present)


def extract_blocks_from_annotations(raw: mne.io.BaseRaw) -> List[Tuple[str, float, float]]:
    """
    Return list of (block_id, start_s, end_s).
    Requires annotations containing B1_START/B1_END etc.
    """
    ann = raw.annotations
    # Collect block boundaries
    starts = {}
    ends = {}
    for onset, desc in zip(ann.onset, ann.description):
        if desc.endswith("_START") and desc.startswith("B"):
            starts[desc.replace("_START", "")] = float(onset)
        if desc.endswith("_END") and desc.startswith("B"):
            ends[desc.replace("_END", "")] = float(onset)

    blocks = []
    for b, s in starts.items():
        if b in ends and ends[b] > s:
            blocks.append((b, s, ends[b]))
    return sorted(blocks, key=lambda x: x[1])


def segment_states(raw: mne.io.BaseRaw) -> List[StateSegment]:
    """
    Convert state-change annotations (T0/T1/T2) into continuous labeled segments.

    Assumption:
      - Each state marker indicates "from now on, state=X until next marker".
    """
    ann = raw.annotations
    # Create list of (onset, desc) sorted
    pairs = sorted([(float(o), str(d)) for o, d in zip(ann.onset, ann.description)], key=lambda x: x[0])

    # Identify state markers only
    state_pairs = [(o, d) for o, d in pairs if d in (REST_MARKER,) + MI_MARKERS]
    if len(state_pairs) < 2:
        raise RuntimeError("Not enough state markers (T0/T1/T2) found in annotations to build segments.")

    # Optional block mapping
    blocks = extract_blocks_from_annotations(raw)  # may be empty
    def block_of_time(t: float) -> Optional[str]:
        for b, s, e in blocks:
            if s <= t <= e:
                return b
        return None

    segs: List[StateSegment] = []
    for (t0, s0), (t1, _) in zip(state_pairs[:-1], state_pairs[1:]):
        if t1 <= t0:
            continue
        label = "rest" if s0 == REST_MARKER else "mi"
        segs.append(StateSegment(onset=t0, offset=t1, label=label, block=block_of_time(t0)))

    return segs


def windows_from_segments(
    segs: List[StateSegment],
    win_len: float,
    win_step: float,
    guard: float,
    train_blocks: Optional[Tuple[str, ...]] = None
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Produce (windows, y) where windows are (start, end) in seconds (raw time),
    and y are 0=rest, 1=mi.
    """
    windows: List[Tuple[float, float]] = []
    labels: List[int] = []

    for seg in segs:
        if train_blocks is not None:
            # blocks come in as "B1","B2","B3"
            if seg.block is None or seg.block not in train_blocks:
                continue

        start = seg.onset + guard
        end = seg.offset
        if end - start < win_len:
            continue

        t = start
        while t + win_len <= end:
            windows.append((t, t + win_len))
            labels.append(0 if seg.label == "rest" else 1)
            t += win_step

    y = np.asarray(labels, dtype=int)
    return windows, y


def build_epochs_from_windows(raw: mne.io.BaseRaw, windows: List[Tuple[float, float]], picks: np.ndarray) -> mne.Epochs:
    """
    Convert time windows into MNE Epochs by creating a fake events array at each window start.
    """
    sfreq = raw.info["sfreq"]
    onsets_samples = np.array([int(round(t0 * sfreq)) for (t0, _) in windows], dtype=int)
    events = np.column_stack([onsets_samples, np.zeros(len(onsets_samples), dtype=int), np.ones(len(onsets_samples), dtype=int)])

    # We epoch from 0..WIN_LEN relative to each window start
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"win": 1},
        tmin=0.0,
        tmax=WIN_LEN,
        baseline=None,
        preload=True,
        picks=picks,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs


# -----------------------
# Main
# -----------------------

def main():
    # Pick input file
    if RAW_FIF_PATH is not None:
        fif_path = Path(RAW_FIF_PATH)
    elif AUTO_PICK_NEWEST:
        fif_path = newest_fif(RECORDINGS_DIR, SUBJECT_FILTER, SESSIONTAG_FILTER)
    else:
        raise ValueError("Set RAW_FIF_PATH or enable AUTO_PICK_NEWEST.")

    print(f"Loading: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # Optional rename channels (later you can map EEG01->C3 etc.)
    apply_channel_rename(raw, CHANNEL_RENAME_MAP)

    picks, picks_desc = resolve_picks(raw)
    print(f"Channel group: {CHANNEL_GROUP} -> {picks_desc}")

    # Basic preprocessing
    if NOTCH is not None:
        raw.notch_filter(NOTCH, picks=picks, verbose=False)
    raw.filter(FREQ_BAND[0], FREQ_BAND[1], picks=picks, verbose=False)

    if RESAMPLE_SFREQ is not None:
        raw.resample(RESAMPLE_SFREQ, npad="auto")

    # Build labeled segments and windows
    segs = segment_states(raw)
    blocks_present = sorted(set([s.block for s in segs if s.block is not None]))
    print(f"Found {len(segs)} state segments. Blocks present: {blocks_present if blocks_present else 'None'}")

    windows, y = windows_from_segments(
        segs,
        win_len=WIN_LEN,
        win_step=WIN_STEP,
        guard=TRANSITION_GUARD,
        train_blocks=TRAIN_BLOCKS,
    )
    if len(windows) < 50:
        raise RuntimeError(f"Too few training windows ({len(windows)}). Check markers, guard, WIN_LEN/STEP.")

    print(f"Extracted windows: {len(windows)} (REST={int((y==0).sum())}, MI={int((y==1).sum())})")

    # Convert into epochs and data
    epochs = build_epochs_from_windows(raw, windows, picks=picks)
    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    # CSP + LDA pipeline
    csp = CSP(n_components=CSP_COMPONENTS, reg=CSP_REG, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()

    clf = Pipeline([
        ("csp", csp),
        ("lda", lda),
    ])

    # Quick CV sanity check
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"CV accuracy ({CV_FOLDS}-fold): {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit final model on all windows
    clf.fit(X, y)

    # Save model + training config (so you know how it was built)
    out_path = MODEL_DIR / MODEL_NAME
    payload = {
        "model": clf,
        "config": {
            "fif_path": str(fif_path),
            "freq_band": FREQ_BAND,
            "notch": NOTCH,
            "resample_sfreq": RESAMPLE_SFREQ,
            "win_len": WIN_LEN,
            "win_step": WIN_STEP,
            "transition_guard": TRANSITION_GUARD,
            "train_blocks": TRAIN_BLOCKS,
            "csp_components": CSP_COMPONENTS,
            "csp_reg": CSP_REG,
            "labels": {"rest": REST_MARKER, "mi": list(MI_MARKERS)},
            "channel_group": CHANNEL_GROUP,
            "channel_picks_desc": picks_desc,
        },
    }
    joblib.dump(payload, out_path)
    print(f"Saved model: {out_path}")


if __name__ == "__main__":
    main()