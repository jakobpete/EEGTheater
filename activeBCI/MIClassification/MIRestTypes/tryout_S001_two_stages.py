# tryout_S002_two_stage.py
#
# Two-stage tryout on a NEW subject (S002):
#   Stage 1: MI vs Rest detector (loaded model)
#   Stage 2: MI type (fists vs feet) classifier (loaded model)
#
# Runs on EEGMMIDB Task 4 imagery runs:
#   R06/R10/R14 (imagined both fists vs both feet)
#
# Expected data location:
#   Pipline/ExampleData/S002/S002R06.edf etc.

import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix


# -----------------------
# Config
# -----------------------
VERSION = "2026-01-27_metrics_v1"
SUBJECT = "S001"
RUNS = [6, 10, 14]

# Filtering should match training as closely as possible
FREQ_BAND = (7.0, 30.0)
NOTCH = 60.0

# Sliding window parameters (should match how you trained/evaluated)
WIN_LEN = 3.0
WIN_STEP = 0.25

# Stage-1 gate threshold: MI detector probability must exceed this to run type output
MI_GATE_THRESH = 0.5

# Optional EMA smoothing for display stability
EMA_ALPHA_MI = 0.10      # stage-1 MI probability smoothing
EMA_ALPHA_TYPE = 0.10    # stage-2 type probability smoothing

# Epoch window definition for ground-truth shading only
# (must match your MI-type script for correct GT shading)
TMIN = 1.0
TMAX = 4.0

RNG_SEED = 42


# -----------------------
# Paths
# -----------------------
# This file will live in: Pipline/MIClassification/MITypes/
PIPELINE_DIR = Path(__file__).resolve().parents[2]  # .../Pipline
DATA_DIR = PIPELINE_DIR / "ExampleData" / SUBJECT

# >>> IMPORTANT: Set these paths to your saved FINAL models <<<
# 1) Stage-1 MI vs Rest model (CSP/LDA or whatever you saved)
MI_VS_REST_MODEL_PATH = PIPELINE_DIR / "MIClassification" / "MIRest" / "models" / "S001_MI_REST_FINAL_ALLRUNS_CSP_LDA.joblib"

# 2) Stage-2 MI types model (fists vs feet)
MI_TYPES_MODEL_PATH = PIPELINE_DIR / "MIClassification" / "MITypes" / "models" / "S001_MI_TYPES_FISTS_FEET_FINAL_ALLRUNS_CSP_LDA.joblib"


# -----------------------
# Helpers
# -----------------------
def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = float(x[0])
    for i in range(1, len(x)):
        y[i] = (1 - alpha) * y[i - 1] + alpha * float(x[i])
    return y


def load_and_preprocess_raw(edf_path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.filter(FREQ_BAND[0], FREQ_BAND[1], fir_design="firwin")
    raw.notch_filter([NOTCH])
    return raw


def contiguous_intervals(mask: np.ndarray):
    m = mask.astype(int)
    d = np.diff(np.r_[0, m, 0])
    ons = np.where(d == 1)[0]
    offs = np.where(d == -1)[0]
    return list(zip(ons, offs))


def build_gt_masks_fists_feet(raw: mne.io.BaseRaw):
    """Ground-truth masks for shading only: T1=fists, T2=feet, window [TMIN, TMAX]."""
    sfreq = raw.info["sfreq"]
    n_samp = raw.n_times
    gt_fists = np.zeros(n_samp, dtype=bool)
    gt_feet = np.zeros(n_samp, dtype=bool)

    events, event_id = mne.events_from_annotations(raw)
    if not all(k in event_id for k in ["T1", "T2"]):
        return gt_fists, gt_feet  # no GT possible

    def mark(mask, start_sec, end_sec):
        s0 = int(np.clip(start_sec * sfreq, 0, n_samp - 1))
        s1 = int(np.clip(end_sec * sfreq, 0, n_samp - 1))
        if s1 > s0:
            mask[s0:s1] = True

    for samp, _, code in events:
        t0 = samp / sfreq
        if code == event_id["T1"]:
            mark(gt_fists, t0 + TMIN, t0 + TMAX)
        elif code == event_id["T2"]:
            mark(gt_feet, t0 + TMIN, t0 + TMAX)

    return gt_fists, gt_feet


def build_gt_mask_mi(raw: mne.io.BaseRaw) -> np.ndarray:
    """Ground-truth MI mask (boolean per sample) based on union of T1/T2 blocks."""
    gt_fists, gt_feet = build_gt_masks_fists_feet(raw)
    return gt_fists | gt_feet


def window_label_from_masks(starts: np.ndarray, win_samp: int, gt_mi: np.ndarray, gt_fists: np.ndarray, gt_feet: np.ndarray,
                           frac_mi: float = 0.5):
    """Return window-wise labels.

    y_mi: 1 if >= frac_mi of samples are MI, else 0.
    y_type: -1 for non-MI windows; 0 for fists; 1 for feet (based on majority overlap within MI windows).
    """
    y_mi = np.zeros(len(starts), dtype=int)
    y_type = -1 * np.ones(len(starts), dtype=int)

    for i, s in enumerate(starts):
        e = s + win_samp
        seg_mi = gt_mi[s:e]
        if seg_mi.size == 0:
            continue
        mi_frac = float(np.mean(seg_mi))
        if mi_frac >= frac_mi:
            y_mi[i] = 1
            # type label by overlap
            f = float(np.mean(gt_fists[s:e]))
            ft = float(np.mean(gt_feet[s:e]))
            y_type[i] = 1 if (ft > f) else 0

    return y_mi, y_type


def simulate_two_stage(raw: mne.io.BaseRaw, mi_model, type_model, title: str, do_plot: bool = True):
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    data = raw.get_data(picks=picks)

    win_samp = int(WIN_LEN * sfreq)
    step_samp = int(WIN_STEP * sfreq)

    starts = np.arange(0, raw.n_times - win_samp, step_samp, dtype=int)
    mid_times = (starts + win_samp / 2) / sfreq

    X_win = np.zeros((len(starts), len(picks), win_samp), dtype=np.float32)
    for i, s in enumerate(starts):
        X_win[i] = data[:, s:s + win_samp]

    # --- Stage 1: MI vs Rest ---
    # Assumption: binary predict_proba gives column 1 = "MI"
    proba_mi = mi_model.predict_proba(X_win)[:, 1]
    proba_mi = ema(proba_mi, EMA_ALPHA_MI)

    # --- Stage 2: MI types (fists vs feet) ---
    # Only meaningful when stage-1 says MI; we still compute it for convenience,
    # but we will gate the visualization and any decisions.
    # Assumption: column 1 = "feet"
    proba_feet = type_model.predict_proba(X_win)[:, 1]
    proba_feet = ema(proba_feet, EMA_ALPHA_TYPE)

    # Signed output: -1=fists, +1=feet. Gated by MI probability.
    signed_type = 2.0 * proba_feet - 1.0
    signed_type_gated = signed_type.copy()
    signed_type_gated[proba_mi < MI_GATE_THRESH] = 0.0

    # -----------------
    # Ground truth labels (from annotations)
    # -----------------
    gt_fists, gt_feet = build_gt_masks_fists_feet(raw)
    gt_mi = gt_fists | gt_feet

    y_true_mi, y_true_type = window_label_from_masks(
        starts=starts,
        win_samp=win_samp,
        gt_mi=gt_mi,
        gt_fists=gt_fists,
        gt_feet=gt_feet,
        frac_mi=0.5,
    )

    # -----------------
    # Stage-1 metrics (MI vs Rest) on windows
    # -----------------
    y_pred_mi = (proba_mi >= MI_GATE_THRESH).astype(int)
    cm_mi = confusion_matrix(y_true_mi, y_pred_mi, labels=[0, 1])
    tn, fp, fn, tp = cm_mi.ravel()
    prec_mi = tp / max(1, (tp + fp))
    rec_mi = tp / max(1, (tp + fn))

    duration_min = raw.times[-1] / 60.0
    fp_per_min = fp / max(1e-9, duration_min)

    # -----------------
    # Stage-2 metrics (type) on MI windows only, gated by stage-1
    # -----------------
    mi_win_idx = np.where(y_true_mi == 1)[0]
    gated_idx = np.where((y_true_mi == 1) & (y_pred_mi == 1))[0]

    type_acc_gated = float("nan")
    type_cm_gated = None
    coverage = 0.0

    if len(mi_win_idx) > 0:
        coverage = float(len(gated_idx) / len(mi_win_idx))

    if len(gated_idx) > 0:
        y_pred_type = (proba_feet >= 0.5).astype(int)
        y_true_type_g = y_true_type[gated_idx]
        y_pred_type_g = y_pred_type[gated_idx]
        type_cm_gated = confusion_matrix(y_true_type_g, y_pred_type_g, labels=[0, 1])
        type_acc_gated = float(np.mean(y_true_type_g == y_pred_type_g))

    if do_plot:
        fig, ax = plt.subplots(figsize=(15, 6))

        # Shade GT blocks
        for s0, s1 in contiguous_intervals(gt_fists):
            ax.axvspan(s0 / sfreq, s1 / sfreq, alpha=0.12)
        for s0, s1 in contiguous_intervals(gt_feet):
            ax.axvspan(s0 / sfreq, s1 / sfreq, alpha=0.22)

        # Plot MI probability (stage 1)
        ax.plot(mid_times, proba_mi, linewidth=1.6, label="Stage 1: p(MI)")

        # Plot gated MI-type signed output (stage 2)
        ax.plot(mid_times, signed_type_gated, linewidth=1.6, label="Stage 2: signed type (gated)")

        ax.axhline(MI_GATE_THRESH, linestyle="--", linewidth=1, label="MI gate threshold")
        ax.axhline(0.0, linestyle="--", linewidth=1)

        ax.set_title(
            title
            + f" | win={WIN_LEN}s step={WIN_STEP}s | gate={MI_GATE_THRESH}"
            + f" | MI prec={prec_mi:.2f} rec={rec_mi:.2f} FP/min={fp_per_min:.2f}"
            + f" | type_acc(gated)={type_acc_gated:.2f} cov={coverage:.2f}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("p(MI) and signed type (-1 fists, +1 feet, 0 none)")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="upper right")

        ax.text(
            0.01, 0.98,
            "Shading: light=fists(T1), dark=feet(T2) using [TMIN,TMAX]\n"
            "Stage-1: p(MI) vs GT-MI (from T1/T2 blocks)\n"
            "Stage-2: signed type is gated (0 when p(MI) < threshold)",
            transform=ax.transAxes,
            va="top",
        )

        plt.tight_layout()
        plt.show()

    print(f"{title} | MI cm [[tn fp],[fn tp]] = {cm_mi.tolist()} | prec={prec_mi:.3f} rec={rec_mi:.3f} fp/min={fp_per_min:.3f}")
    if type_cm_gated is not None:
        print(f"{title} | TYPE gated cm rows=true [fists,feet] cols=pred [fists,feet]: {type_cm_gated.tolist()} | acc={type_acc_gated:.3f} | coverage={coverage:.3f}")
    else:
        print(f"{title} | TYPE gated: no gated MI windows (coverage={coverage:.3f})")

    return {
        "n_windows": len(starts),
        "mi_gate_thresh": MI_GATE_THRESH,
        "mi_rate": float(np.mean(proba_mi >= MI_GATE_THRESH)),
        "mi_cm": cm_mi.tolist(),
        "mi_precision": float(prec_mi),
        "mi_recall": float(rec_mi),
        "mi_fp_per_min": float(fp_per_min),
        "type_acc_gated": float(type_acc_gated) if not np.isnan(type_acc_gated) else None,
        "type_cm_gated": type_cm_gated.tolist() if type_cm_gated is not None else None,
        "type_coverage": float(coverage),
    }


def main():
    print(f"Running: {__file__}")
    print(f"Version: {VERSION}")
    # Load models
    if not MI_VS_REST_MODEL_PATH.exists():
        raise FileNotFoundError(f"MI-vs-Rest model not found: {MI_VS_REST_MODEL_PATH}")
    if not MI_TYPES_MODEL_PATH.exists():
        raise FileNotFoundError(f"MI-Types model not found: {MI_TYPES_MODEL_PATH}")

    mi_model = joblib.load(MI_VS_REST_MODEL_PATH)
    type_model = joblib.load(MI_TYPES_MODEL_PATH)

    print("Loaded stage-1 model:", MI_VS_REST_MODEL_PATH)
    print("Loaded stage-2 model:", MI_TYPES_MODEL_PATH)

    # Run each Task-4 file for S002
    for r in RUNS:
        edf_path = DATA_DIR / f"{SUBJECT}R{r:02d}.edf"
        if not edf_path.exists():
            print("Missing EDF:", edf_path)
            continue

        raw = load_and_preprocess_raw(edf_path)
        title = f"{SUBJECT} | R{r:02d} two-stage MI → type (fists vs feet)"
        stats = simulate_two_stage(raw, mi_model, type_model, title=title, do_plot=True)
        print(f"Run R{r:02d} stats: {stats}")


if __name__ == "__main__":
    main()