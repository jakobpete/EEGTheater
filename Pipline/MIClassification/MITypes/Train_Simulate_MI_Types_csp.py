# Uses:
#   CSP + LDA
#
# Evaluates properly with leave-one-run-out (train on 2 runs, test on 1 run),
# and includes an "online-like" sliding window simulation restricted to MI segments.
#
# Saves final deployment model trained on ALL runs in:
#   Pipline/MIClassification/MITypes/models/

import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

# -----------------------
# Config
# -----------------------
SUBJECT = "S002"

# Imagined BOTH FISTS vs BOTH FEET runs in EEGMMIDB:
# R06/R10/R14 = Task 4 (imagine both fists vs both feet)
RUNS = [6, 10, 14]

# Preprocessing
FREQ_BAND = (8.0, 20.0) # high impact!!!!
NOTCH = 60.0

# Set global number of componants for GSP
CSP_COMPONENTS = 4

# -----------------------
# Channel selection
# -----------------------
# "all"           : use all EEG channels
# "distributed16" : 16 channels distributed across scalp
# "motor_roi"     : motor-area channels (C3/Cz/C4 + neighbors)
CHANNEL_GROUP = "all"

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

DISTRIBUTED_16_INDEX_FALLBACK = [0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]
MOTOR_ROI_INDEX_FALLBACK = [6,7,8,14,15,16,22,23,24]

# Epoch window after event markers
TMIN = 1.0
TMAX = 4.0

# Online simulation windowing
WIN_LEN = 3.0
WIN_STEP = 0.25

# For “which MI type” we threshold only for display; classification is argmax
THRESH = 0.5

EMA_ALPHA_CSP = 0.00

# Block scoring overlap
MIN_OVERLAP_S = 1.0

RNG_SEED = 42

# -----------------------
# Paths
# -----------------------
# This file lives in: Pipline/MIClassification/MITypes/
PIPELINE_DIR = Path(__file__).resolve().parents[2]  # .../Pipline
DATA_DIR = PIPELINE_DIR / "ExampleData" / SUBJECT

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def load_and_preprocess_raw(edf_path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    picks, picks_desc = resolve_picks(raw)
    raw.filter(FREQ_BAND[0], FREQ_BAND[1], fir_design="firwin", picks=picks)
    raw.notch_filter([NOTCH], picks=picks)
    raw.info["_custom_picks"] = picks
    raw.info["_custom_picks_desc"] = picks_desc
    return raw


# -----------------------
# Channel group helper
# -----------------------
def resolve_picks(raw: mne.io.BaseRaw):
    """Resolve channel indices based on CHANNEL_GROUP."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude="bads")

    if CHANNEL_GROUP == "all":
        return eeg_picks, f"all ({len(eeg_picks)} EEG channels)"

    def name_based(names):
        want = [n for n in names if n in raw.ch_names]
        if not want:
            return None
        return mne.pick_channels(raw.ch_names, include=want)

    if CHANNEL_GROUP == "distributed16":
        p = name_based(DISTRIBUTED_16_NAMES)
        if p is not None and len(p) >= 8:
            return p, f"distributed16 (name-based, n={len(p)})"
        idx = [i for i in DISTRIBUTED_16_INDEX_FALLBACK if i < len(eeg_picks)]
        return eeg_picks[idx], f"distributed16 (index-fallback, n={len(idx)})"

    if CHANNEL_GROUP == "motor_roi":
        p = name_based(MOTOR_ROI_NAMES)
        if p is not None and len(p) >= 3:
            return p, f"motor_roi (name-based, n={len(p)})"
        idx = [i for i in MOTOR_ROI_INDEX_FALLBACK if i < len(eeg_picks)]
        return eeg_picks[idx], f"motor_roi (index-fallback, n={len(idx)})"

    raise ValueError(f"Unknown CHANNEL_GROUP='{CHANNEL_GROUP}'")




def build_epochs_fists_vs_feet(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Build balanced T1 vs T2 epochs for Task 4 imagery runs (R06/R10/R14).
    In these runs:
      T1 = both fists (imagined)
      T2 = both feet (imagined)
    """
    events, event_id = mne.events_from_annotations(raw)
    for code in ["T1", "T2"]:
        if code not in event_id:
            raise KeyError(f"Missing {code} in event_id={event_id}")

    fists_code = event_id["T1"]
    feet_code = event_id["T2"]

    ev_fists = events[events[:, 2] == fists_code].copy()
    ev_feet = events[events[:, 2] == feet_code].copy()

    # Recode: fists=1, feet=2 (binary)
    ev_fists[:, 2] = 1
    ev_feet[:, 2] = 2

    rng = np.random.default_rng(RNG_SEED)
    n_f, n_fe = len(ev_fists), len(ev_feet)
    if n_f == 0 or n_fe == 0:
        raise ValueError(f"No events: fists={n_f}, feet={n_fe}")

    # Balance within run
    if n_f > n_fe:
        idx = rng.choice(n_f, size=n_fe, replace=False)
        ev_fists = ev_fists[idx]
    elif n_fe > n_f:
        idx = rng.choice(n_fe, size=n_f, replace=False)
        ev_feet = ev_feet[idx]

    ev_bin = np.vstack([ev_fists, ev_feet])
    ev_bin = ev_bin[np.argsort(ev_bin[:, 0])]

    epochs = mne.Epochs(
        raw,
        ev_bin,
        event_id={"fists": 1, "feet": 2},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs




def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = float(x[0])
    for i in range(1, len(x)):
        y[i] = (1 - alpha) * y[i - 1] + alpha * float(x[i])
    return y


def contiguous_intervals(mask: np.ndarray):
    m = mask.astype(int)
    d = np.diff(np.r_[0, m, 0])
    ons = np.where(d == 1)[0]
    offs = np.where(d == -1)[0]
    return list(zip(ons, offs))


def build_gt_masks_fists_feet(raw: mne.io.BaseRaw):
    """Return sample-level ground-truth masks for fists and feet MI segments."""
    sfreq = raw.info["sfreq"]
    n_samp = raw.n_times
    gt_fists = np.zeros(n_samp, dtype=bool)
    gt_feet = np.zeros(n_samp, dtype=bool)

    events, event_id = mne.events_from_annotations(raw)
    if not all(k in event_id for k in ["T1", "T2"]):
        raise KeyError(f"Missing T1/T2 in event_id={event_id}")

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


def simulate_online_on_raw_types(raw: mne.io.BaseRaw, csp_clf, title: str, do_plot: bool = True):
    """Sliding-window simulation on a run, but scoring only on windows that are inside MI segments."""
    sfreq = raw.info["sfreq"]
    picks = raw.info.get("_custom_picks")
    data = raw.get_data(picks=picks)

    n_samp = raw.n_times
    gt_fists, gt_feet = build_gt_masks_fists_feet(raw)
    gt_any = gt_fists | gt_feet

    win_samp = int(WIN_LEN * sfreq)
    step_samp = int(WIN_STEP * sfreq)

    starts = np.arange(0, n_samp - win_samp, step_samp, dtype=int)
    mid_times = (starts + win_samp / 2) / sfreq

    # window tensors
    X_win = np.zeros((len(starts), len(picks), win_samp), dtype=np.float32)

    # window labels: -1 outside MI, 0=fists, 1=feet for MI windows
    y_true = np.full(len(starts), -1, dtype=int)

    for i, s in enumerate(starts):
        X_win[i] = data[:, s:s + win_samp]

        frac_any = gt_any[s:s + win_samp].mean()
        if frac_any < 0.5:
            continue  # ignore non-MI windows

        frac_f = gt_fists[s:s + win_samp].mean()
        frac_fe = gt_feet[s:s + win_samp].mean()
        y_true[i] = 0 if frac_f >= frac_fe else 1

    mi_idx = np.where(y_true >= 0)[0]
    if len(mi_idx) == 0:
        raise RuntimeError("No MI windows found for scoring. Check RUNS/TMIN/TMAX/WIN_LEN.")

    # ---- CSP probabilities (binary) ----
    proba_csp = csp_clf.predict_proba(X_win)  # shape (n_win, 2) with columns [p(fists), p(feet)]

    # IMPORTANT: smooth ONLY one probability to keep normalization (p0 + p1 = 1)
    p_csp_feet = ema(proba_csp[:, 1], EMA_ALPHA_CSP)
    p_csp_feet = np.clip(p_csp_feet, 0.0, 1.0)
    p_csp_fists = 1.0 - p_csp_feet
    proba_csp_sm = np.c_[p_csp_fists, p_csp_feet]
    y_pred_csp = np.argmax(proba_csp_sm, axis=1)

    # Signed output for visualization:
    #   -1 = fists, +1 = feet, 0 = no-MI (outside MI windows)
    signed_csp = 2.0 * proba_csp_sm[:, 1] - 1.0  # map p(feet) in [0,1] to [-1, +1]
    signed_csp_vis = signed_csp.copy()
    signed_csp_vis[y_true < 0] = 0.0

    # Window accuracy computed only on MI windows
    acc_csp = float((y_pred_csp[mi_idx] == y_true[mi_idx]).mean())

    # ---- Block-level confusion matrix and accuracy (per MI block): majority prediction within each true block ----
    def block_confusion(p_feet):
        """Block-level confusion matrix.
        Rows=true [fists, feet], cols=pred [fists, feet].
        A block is labeled by mean p(feet) over overlapping windows.
        """
        blocks = []
        for s0, s1 in contiguous_intervals(gt_fists):
            blocks.append((s0, s1, 0))
        for s0, s1 in contiguous_intervals(gt_feet):
            blocks.append((s0, s1, 1))
        blocks = sorted(blocks, key=lambda x: x[0])

        cm = np.zeros((2, 2), dtype=int)
        min_ov = int(MIN_OVERLAP_S * sfreq)

        for bs, be, true_lab in blocks:
            ov_idx = []
            for i, s in enumerate(starts):
                e = s + win_samp
                ov = min(be, e) - max(bs, s)
                if ov >= min_ov:
                    ov_idx.append(i)
            if not ov_idx:
                continue
            # Block label by mean p(feet) over overlapping windows
            pred_lab = 1 if (np.mean(p_feet[ov_idx]) >= 0.5) else 0
            cm[true_lab, pred_lab] += 1

        return cm

    def block_type_accuracy(p_feet):
        cm = block_confusion(p_feet)
        total = int(cm.sum())
        acc = (cm[0, 0] + cm[1, 1]) / total if total else float("nan")
        return float(acc), total, cm

    block_acc_csp, n_blocks, cm_block_csp = block_type_accuracy(proba_csp_sm[:, 1])

    # MI-window class balance (for sanity)
    n_fists_win = int(np.sum(y_true[mi_idx] == 0))
    n_feet_win = int(np.sum(y_true[mi_idx] == 1))

    # Confusion matrices on MI windows only
    cm_csp = confusion_matrix(y_true[mi_idx], y_pred_csp[mi_idx], labels=[0, 1])

    csp_acc_fists = cm_csp[0, 0] / max(1, (cm_csp[0, 0] + cm_csp[0, 1]))
    csp_acc_feet = cm_csp[1, 1] / max(1, (cm_csp[1, 0] + cm_csp[1, 1]))

    print(
        f"{title}: CSP window-acc (MI windows only)={acc_csp:.3f} | block-acc={block_acc_csp:.3f} (n_blocks={n_blocks}) | "
        f"MI-win balance fists={n_fists_win} feet={n_feet_win} | per-class acc fists={csp_acc_fists:.3f} feet={csp_acc_feet:.3f}"
    )

    if do_plot:
        fig, ax = plt.subplots(figsize=(14, 5))

        # shade true fists and feet segments differently
        # fists: light shading, feet: darker shading
        for s0, s1 in contiguous_intervals(gt_fists):
            ax.axvspan(s0 / sfreq, s1 / sfreq, alpha=0.15)
        for s0, s1 in contiguous_intervals(gt_feet):
            ax.axvspan(s0 / sfreq, s1 / sfreq, alpha=0.30)

        # Plot signed CSP output: -1=fists, +1=feet, 0=no-MI
        ax.plot(mid_times, signed_csp_vis, linewidth=1.8, label="CSP signed output")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axhline(+0.5, linestyle=":", linewidth=1)
        ax.axhline(-0.5, linestyle=":", linewidth=1)

        ax.set_title(title + f" | win={WIN_LEN}s step={WIN_STEP}s | MI-win acc: CSP={acc_csp:.3f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("signed output (-1 fists, +1 feet, 0 no-MI)")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="upper right")

        ax.text(
            0.01, 0.95,
            "Shaded: light=fists MI, dark=feet MI (T1/T2 + [TMIN,TMAX])\n"
            "Line: signed CSP output (0 outside MI windows; -1=fists, +1=feet)",
            transform=ax.transAxes,
            va="top",
        )
        plt.tight_layout()
        plt.show()

        # --- Histogram of signed CSP values during MI windows, split by true class ---
        signed_mi = signed_csp[mi_idx]
        true_mi = y_true[mi_idx]  # 0=fists, 1=feet

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(signed_mi[true_mi == 0], bins=25, alpha=0.6, density=True, label="True fists")
        ax2.hist(signed_mi[true_mi == 1], bins=25, alpha=0.6, density=True, label="True feet")
        ax2.axvline(0.0, linestyle="--", linewidth=1)
        ax2.set_title(title + " | CSP signed output distribution (MI windows)")
        ax2.set_xlabel("signed output (-1 fists, +1 feet)")
        ax2.set_ylabel("density")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        # --- Simple correctness bars + confusion matrix (MI windows only) ---
        cm = cm_csp
        correct = int(cm[0, 0] + cm[1, 1])
        wrong = int(cm[0, 1] + cm[1, 0])
        acc_fists = cm[0, 0] / max(1, (cm[0, 0] + cm[0, 1]))
        acc_feet = cm[1, 1] / max(1, (cm[1, 0] + cm[1, 1]))

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(["correct", "wrong"], [correct, wrong])
        ax3.set_title(title + f" | CSP correctness (MI windows) | overall acc={acc_csp:.3f}")
        ax3.set_ylabel("# windows")
        plt.tight_layout()
        plt.show()

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.bar(["fists", "feet"], [acc_fists, acc_feet])
        ax4.set_ylim(0, 1)
        ax4.set_title(title + " | CSP per-class accuracy (MI windows)")
        ax4.set_ylabel("accuracy")
        plt.tight_layout()
        plt.show()

        print("Confusion matrix (MI windows only), rows=true [fists, feet], cols=pred [fists, feet]:")
        print(cm)

        # --- Block-level confusion matrix summary ---
        print("Block confusion (CSP), rows=true [fists, feet], cols=pred [fists, feet]:")
        print(cm_block_csp)

        fig5, ax5 = plt.subplots(figsize=(7, 4))
        block_correct = int(cm_block_csp[0, 0] + cm_block_csp[1, 1])
        block_wrong = int(cm_block_csp[0, 1] + cm_block_csp[1, 0])
        ax5.bar(["block correct", "block wrong"], [block_correct, block_wrong])
        ax5.set_title(title + f" | CSP block correctness (n_blocks={n_blocks}) | block acc={block_acc_csp:.3f}")
        ax5.set_ylabel("# blocks")
        plt.tight_layout()
        plt.show()

    return {
        "window_acc_csp": acc_csp,
        "block_acc_csp": float(block_acc_csp),
        "cm_csp": cm_csp.tolist(),
        "cm_block_csp": cm_block_csp.tolist(),
    }


# -----------------------
# Load data
# -----------------------
raw_by_run = {}
for r in RUNS:
    edf_path = DATA_DIR / f"{SUBJECT}R{r:02d}.edf"
    print("Loading EDF:", edf_path)
    raw_by_run[r] = load_and_preprocess_raw(edf_path)

# -----------------------
# Leave-one-run-out evaluation
# -----------------------
fold_stats = []

for test_run in RUNS:
    train_runs = [r for r in RUNS if r != test_run]

    # Epoch training data from training runs
    train_epochs = [build_epochs_fists_vs_feet(raw_by_run[r]) for r in train_runs]
    epochs_train = mne.concatenate_epochs(train_epochs)

    picks = raw_by_run[train_runs[0]].info.get("_custom_picks")
    X_train = epochs_train.get_data(picks=picks)
    y_train = (epochs_train.events[:, 2] == epochs_train.event_id["feet"]).astype(int)  # fists=0, feet=1

    # CSP + LDA (binary)
    X_train_full = X_train
    csp = CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    csp_clf = Pipeline([("csp", csp), ("lda", lda)])
    csp_clf.fit(X_train_full, y_train)

    # Online-like simulation on held-out run
    do_plot = (test_run == RUNS[-1])
    title = f"Held-out run R{test_run:02d} (trained on {train_runs})"
    stats = simulate_online_on_raw_types(raw_by_run[test_run], csp_clf, title=title, do_plot=do_plot)
    fold_stats.append(stats)

    # Save fold models (optional but useful for debugging)
    prefix = f"{SUBJECT}_MI_TYPES_FISTS_FEET_fold_heldout_R{test_run:02d}"
    joblib.dump(csp_clf, MODEL_DIR / f"{prefix}_CSP_LDA.joblib")

# Summary
mean_csp = float(np.mean([s["window_acc_csp"] for s in fold_stats]))
mean_csp_block = float(np.mean([s["block_acc_csp"] for s in fold_stats]))

print("\n=== Mean over folds (MI windows only) ===")
print(f"Window-acc: CSP={mean_csp:.3f}")
print(f"Block-acc:  CSP={mean_csp_block:.3f}")

# -----------------------
# Train + save FINAL models on ALL runs (deployment)
# -----------------------
all_epochs = [build_epochs_fists_vs_feet(raw_by_run[r]) for r in RUNS]
epochs_all = mne.concatenate_epochs(all_epochs)

picks = raw_by_run[RUNS[0]].info.get("_custom_picks")
X_all = epochs_all.get_data(picks=picks)
y_all = (epochs_all.events[:, 2] == epochs_all.event_id["feet"]).astype(int)

X_all_full = X_all

final_csp = CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)
final_lda = LinearDiscriminantAnalysis()
final_csp_lda = Pipeline([("csp", final_csp), ("lda", final_lda)])
final_csp_lda.fit(X_all_full, y_all)

final_prefix = f"{SUBJECT}_MI_TYPES_FISTS_FEET_FINAL_ALLRUNS_{CHANNEL_GROUP}"
joblib.dump(final_csp_lda, MODEL_DIR / f"{final_prefix}_CSP_LDA.joblib")

meta = {
    "subject": SUBJECT,
    "runs": [int(r) for r in RUNS],
    "task": "imagined fists vs imagined feet (R06/R10/R14)",
    "labels": {"fists": 0, "feet": 1},
    "freq_band": FREQ_BAND,
    "notch": NOTCH,
    "tmin": TMIN,
    "tmax": TMAX,
    "win_len": WIN_LEN,
    "win_step": WIN_STEP,
    "channel_group": CHANNEL_GROUP,
    "channel_desc": raw_by_run[RUNS[0]].info.get("_custom_picks_desc"),
}
joblib.dump(meta, MODEL_DIR / f"{final_prefix}_META.joblib")

print("\nSaved FINAL model to:", MODEL_DIR)