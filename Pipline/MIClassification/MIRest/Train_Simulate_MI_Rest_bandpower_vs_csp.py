import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import joblib

# -----------------------
# Config
# -----------------------
SUBJECT = "S001"

# Stage-1 (MI vs Rest) for the fists-vs-feet pipeline must be trained on Task-4 imagery runs:
# R06/R10/R14 = imagine BOTH FISTS (T1) vs BOTH FEET (T2) with rest periods (T0) between trials.
# We pool T1+T2 as MI for the Stage-1 detector.
RUNS = [6, 10, 14]

# Must match your training preprocessing
FREQ_BAND = (8.0, 20.0)
NOTCH = 60.0

# -----------------------
# Channel selection (ONE knob for both CSP + Bandpower)
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

# Fallback indices if channel names don't match (clipped to available channels)
DISTRIBUTED_16_INDEX_FALLBACK = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
MOTOR_ROI_INDEX_FALLBACK = [6, 7, 8, 14, 15, 16, 22, 23, 24]

# Bandpower detector configuration
BANDS = {
    "mu": (8.0, 12.0),
    "beta": (13.0, 20.0),
}

# These define the "meaningful" segment after an imagery onset marker (T1/T2)
# T0 marks rest onsets. We treat windows after T0 as "rest" examples.
TMIN = 1.0
TMAX = 4.0

# CSP COMPONENTS
CSP_COMPONENTS = 4

# Online simulation windowing
WIN_LEN = 3.0      # seconds (try 2.0–3.0)
WIN_STEP = 0.25    # seconds (try 0.1–0.5)
THRESH = 0.5       # probability threshold for MI

# Smoothing (EMA) for probability streams (helps online stability)
EMA_ALPHA_BP = 0.15   # 0.05–0.30 reasonable; higher = more smoothing
# EMA_ALPHA_CSP controls temporal smoothing of the *output probability* of the CSP+LDA classifier.
# IMPORTANT: this does NOT affect training or the classifier itself.
# It only smooths the probability stream over time during online / simulated use.
#
# alpha = 0.0    -> no smoothing (raw CSP probabilities, best for evaluation/debugging)
# alpha ~0.05    -> strong smoothing (very stable but delayed response)
# alpha ~0.1–0.2 -> moderate smoothing (good compromise for online use)
#
# Use smoothing only at runtime (e.g. on stage) to reduce jitter and false positives.
# Keep it at 0.0 when evaluating classifier performance.
EMA_ALPHA_CSP = 0.00  # keep 0.00 to not smooth CSP; set e.g. 0.10 if you want both smoothed

# Block-detection scoring
MIN_OVERLAP_S = 0.25   # minimum overlap (seconds) to count a predicted block as matching a true MI block

# -----------------------
# Paths
# -----------------------
# This file lives in: Pipline/MIClassification/MIRest/
PIPELINE_DIR = Path(__file__).resolve().parents[2]  # .../Pipline
DATA_DIR = PIPELINE_DIR / "ExampleData" / SUBJECT

# Save trained models here: Pipline/MIClassification/MIRest/models
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Helpers
# -----------------------

def print_config():
    print("\n" + "=" * 60)
    print("Running configuration:")
    print("-" * 60)

    print(f"SUBJECT            : {SUBJECT}")
    print(f"RUNS               : {RUNS}")

    print("\nPreprocessing:")
    print(f"  FREQ_BAND         : {FREQ_BAND}")
    print(f"  NOTCH             : {NOTCH}")

    print("\nChannel selection:")
    print(f"  CHANNEL_GROUP     : {CHANNEL_GROUP}")

    print("\nBandpower:")
    print(f"  BANDS             : {BANDS}")

    print("\nEpoching:")
    print(f"  TMIN              : {TMIN}")
    print(f"  TMAX              : {TMAX}")

    print("\nCSP:")
    print(f"  CSP_COMPONENTS    : {CSP_COMPONENTS}")

    print("\nOnline windowing:")
    print(f"  WIN_LEN           : {WIN_LEN}")
    print(f"  WIN_STEP          : {WIN_STEP}")
    print(f"  THRESH            : {THRESH}")

    print("\nSmoothing:")
    print(f"  EMA_ALPHA_BP      : {EMA_ALPHA_BP}")
    print(f"  EMA_ALPHA_CSP     : {EMA_ALPHA_CSP}")

    print("\nBlock detection:")
    print(f"  MIN_OVERLAP_S     : {MIN_OVERLAP_S}")

    print("=" * 60 + "\n")


def load_and_preprocess_raw(edf_path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.filter(FREQ_BAND[0], FREQ_BAND[1], fir_design="firwin")
    raw.notch_filter([NOTCH])
    return raw


def _pick_by_names_or_fallback(ch_names: list[str], wanted_names: list[str], fallback_idx: list[int]) -> np.ndarray:
    """Pick channels by (case-insensitive) name; if too few found, use fallback indices."""
    upper = [c.upper() for c in ch_names]
    wanted_upper = [w.upper() for w in wanted_names]
    picks = [upper.index(w) for w in wanted_upper if w in upper]

    if len(picks) >= 3:
        return np.array(picks, dtype=int)

    n = len(ch_names)
    picks_fb = [i for i in fallback_idx if 0 <= i < n]
    return np.array(picks_fb, dtype=int)


def pick_channels(obj) -> np.ndarray:
    """Return channel picks according to CHANNEL_GROUP.

    This ONE selection is used for BOTH CSP+LDA and Bandpower+LogReg.
    Accepts Raw or Epochs.
    """
    info = obj.info
    ch_names = obj.ch_names

    if CHANNEL_GROUP == "all":
        return mne.pick_types(info, eeg=True, exclude="bads")

    if CHANNEL_GROUP == "distributed16":
        picks = _pick_by_names_or_fallback(ch_names, DISTRIBUTED_16_NAMES, DISTRIBUTED_16_INDEX_FALLBACK)
    elif CHANNEL_GROUP == "motor_roi":
        picks = _pick_by_names_or_fallback(ch_names, MOTOR_ROI_NAMES, MOTOR_ROI_INDEX_FALLBACK)
    else:
        raise ValueError(f"Unknown CHANNEL_GROUP={CHANNEL_GROUP!r} (use 'all', 'distributed16', or 'motor_roi')")

    if picks.size == 0:
        picks = mne.pick_types(info, eeg=True, exclude="bads")

    return picks


def build_epochs_mi_rest(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Build balanced MI-vs-Rest epochs from T0/T1/T2 markers.

    MI = T1 + T2 pooled, Rest = T0.
    Uses window [TMIN, TMAX] after each marker.
    Balances by downsampling the larger class.
    """
    events, event_id = mne.events_from_annotations(raw)
    for code in ["T0", "T1", "T2"]:
        if code not in event_id:
            raise KeyError(f"Missing {code} in event_id={event_id}")

    # Select onsets
    mi_codes = [event_id["T1"], event_id["T2"]]
    rest_code = event_id["T0"]

    events_mi = events[np.isin(events[:, 2], mi_codes)].copy()
    events_rest = events[events[:, 2] == rest_code].copy()

    # Recode binary (rest=1, mi=2)
    events_mi[:, 2] = 2
    events_rest[:, 2] = 1

    # Balance within this run
    rng = np.random.default_rng(42)
    n_mi = len(events_mi)
    n_rest = len(events_rest)
    if n_mi == 0 or n_rest == 0:
        raise ValueError(f"No events to build epochs: n_mi={n_mi}, n_rest={n_rest}")

    if n_rest > n_mi:
        idx = rng.choice(n_rest, size=n_mi, replace=False)
        events_rest = events_rest[idx]
    elif n_mi > n_rest:
        idx = rng.choice(n_mi, size=n_rest, replace=False)
        events_mi = events_mi[idx]
        n_mi = len(events_mi)

    events_bin = np.vstack([events_rest, events_mi])
    events_bin = events_bin[np.argsort(events_bin[:, 0])]

    epochs = mne.Epochs(
        raw,
        events_bin,
        event_id={"rest": 1, "mi": 2},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs


def log_bandpower(window_data: np.ndarray, sfreq: float, band: tuple[float, float]):
    """Compute log bandpower per channel for a window.

    Uses Welch PSD with overlap for lower-variance estimates.
    window_data shape: (n_ch, n_times)
    """
    fmin, fmax = band

    # Welch parameters (stabilize PSD estimates for short windows)
    n_per_seg = min(256, window_data.shape[1])
    n_fft = n_per_seg
    n_overlap = n_per_seg // 2

    psd, freqs = mne.time_frequency.psd_array_welch(
        window_data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )
    bp = np.trapz(psd, freqs, axis=1)
    return np.log(bp + 1e-12)


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing."""
    if alpha <= 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = float(x[0])
    for i in range(1, len(x)):
        y[i] = (1 - alpha) * y[i - 1] + alpha * float(x[i])
    return y


def simulate_online_on_raw(raw: mne.io.BaseRaw, csp_clf, bp_clf, bp_norm, title: str, do_plot: bool = True):
    """Sliding-window simulation on a *single* run."""
    sfreq = raw.info["sfreq"]
    picks = pick_channels(raw)
    data = raw.get_data(picks=picks)

    n_samp = raw.n_times
    # Ground-truth masks: pooled MI and MI-by-type (Task-4 only)
    gt_mi = np.zeros(n_samp, dtype=bool)
    gt_fists = np.zeros(n_samp, dtype=bool)  # T1 in Task-4 runs
    gt_feet = np.zeros(n_samp, dtype=bool)   # T2 in Task-4 runs

    def mark_interval(mask: np.ndarray, start_sec: float, end_sec: float, value: bool = True):
        s0 = int(np.clip(start_sec * sfreq, 0, n_samp - 1))
        s1 = int(np.clip(end_sec * sfreq, 0, n_samp - 1))
        if s1 > s0:
            mask[s0:s1] = value

    events, event_id = mne.events_from_annotations(raw)
    if not all(k in event_id for k in ["T0", "T1", "T2"]):
        raise KeyError(f"Missing expected event codes in event_id={event_id}")

    for ev in events:
        samp, _, code = ev
        t0 = samp / sfreq
        if code == event_id["T1"]:
            # Task-4: T1 = both fists
            mark_interval(gt_mi, t0 + TMIN, t0 + TMAX, True)
            mark_interval(gt_fists, t0 + TMIN, t0 + TMAX, True)
        elif code == event_id["T2"]:
            # Task-4: T2 = both feet
            mark_interval(gt_mi, t0 + TMIN, t0 + TMAX, True)
            mark_interval(gt_feet, t0 + TMIN, t0 + TMAX, True)

    win_samp = int(WIN_LEN * sfreq)
    step_samp = int(WIN_STEP * sfreq)

    starts = np.arange(0, n_samp - win_samp, step_samp, dtype=int)
    mid_times = (starts + win_samp / 2) / sfreq

    X_win = np.zeros((len(starts), len(picks), win_samp), dtype=np.float32)
    y_true = np.zeros(len(starts), dtype=int)

    for i, s in enumerate(starts):
        X_win[i] = data[:, s: s + win_samp]
        frac_mi = gt_mi[s: s + win_samp].mean()
        y_true[i] = 1 if frac_mi >= 0.5 else 0

    # CSP+LDA probabilities
    p_csp = csp_clf.predict_proba(X_win)[:, 1]
    p_csp = ema(p_csp, EMA_ALPHA_CSP)

    # Bandpower probabilities (use SAME channel selection)
    # NOTE: `data` is ALREADY restricted to `picks` (shape = n_sel_ch x n_times),
    # so we must NOT index again with raw-channel indices.
    X_bp_win = []
    for i, s in enumerate(starts):
        seg = data[:, s: s + win_samp]  # (n_sel_ch, win_samp)
        feats = []
        for band in BANDS.values():
            feats.append(log_bandpower(seg, sfreq=sfreq, band=band))
        X_bp_win.append(np.concatenate(feats))
    X_bp_win = np.vstack(X_bp_win)

    # Z-score bandpower features using training-run stats (reduces drift / run-to-run scaling issues)
    bp_mean, bp_std = bp_norm
    X_bp_win = (X_bp_win - bp_mean) / (bp_std + 1e-12)

    p_bp = bp_clf.predict_proba(X_bp_win)[:, 1]

    # Smooth probability stream for online stability
    p_bp = ema(p_bp, EMA_ALPHA_BP)

    # Thresholded predictions (for window accuracy)
    y_pred_csp = (p_csp >= THRESH).astype(int)
    y_pred_bp = (p_bp >= THRESH).astype(int)
    acc_csp = (y_pred_csp == y_true).mean()
    acc_bp = (y_pred_bp == y_true).mean()

    # -----------------------
    # Block-level detection stats
    # -----------------------
    # Create sample-level prediction mask by marking each window as MI across its full duration.

    def block_stats(p, gt_mask: np.ndarray):
        pred_mi = np.zeros(n_samp, dtype=bool)
        for s, prob in zip(starts, p):
            if prob >= THRESH:
                pred_mi[s: s + win_samp] = True

        def contiguous_intervals(mask: np.ndarray):
            """Return list of (start_samp, end_samp) for contiguous True regions."""
            m = mask.astype(int)
            d = np.diff(np.r_[0, m, 0])
            ons = np.where(d == 1)[0]
            offs = np.where(d == -1)[0]
            return list(zip(ons, offs))

        gt_intervals = contiguous_intervals(gt_mask)
        pred_intervals = contiguous_intervals(pred_mi)

        min_ov_samp = int(MIN_OVERLAP_S * sfreq)

        # Match predicted blocks to true blocks by overlap
        tp = 0
        fn = 0
        fp = 0
        latencies = []

        matched_pred = set()

        for (gs, ge) in gt_intervals:
            # Find any predicted interval that overlaps enough
            best_pi = None
            for pi, (ps, pe) in enumerate(pred_intervals):
                ov = min(ge, pe) - max(gs, ps)
                if ov >= min_ov_samp:
                    best_pi = pi
                    break

            if best_pi is None:
                fn += 1
            else:
                tp += 1
                matched_pred.add(best_pi)
                # latency: first predicted sample within the true MI block minus true onset
                ps, _ = pred_intervals[best_pi]
                first_in = max(gs, ps)
                latencies.append((first_in - gs) / sfreq)

        for pi in range(len(pred_intervals)):
            if pi not in matched_pred:
                fp += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        mean_latency = float(np.mean(latencies)) if len(latencies) else float("nan")

        duration_min = (n_samp / sfreq) / 60.0
        fp_per_min = fp / duration_min if duration_min > 0 else float("nan")

        return {
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "n_true_blocks": int(len(gt_intervals)),
            "n_pred_blocks": int(len(pred_intervals)),
            "recall": float(recall),
            "precision": float(precision),
            "fp_per_min": float(fp_per_min),
            "mean_latency_s": float(mean_latency),
        }

    # Pooled MI stats
    stats_csp = block_stats(p_csp, gt_mi)
    stats_bp = block_stats(p_bp, gt_mi)

    # Per-type MI stats (Task-4 only): how often the detector catches fists vs feet blocks
    stats_csp_fists = block_stats(p_csp, gt_fists)
    stats_csp_feet = block_stats(p_csp, gt_feet)
    stats_bp_fists = block_stats(p_bp, gt_fists)
    stats_bp_feet = block_stats(p_bp, gt_feet)

    print(
        f"{title}: CSP+LDA win_acc={acc_csp:.3f} | blocks TP={stats_csp['tp']}/{stats_csp['n_true_blocks']} FN={stats_csp['fn']} FP={stats_csp['fp']} "
        f"| recall={stats_csp['recall']:.3f} (fists={stats_csp_fists['recall']:.3f}, feet={stats_csp_feet['recall']:.3f}) "
        f"precision={stats_csp['precision']:.3f} FP/min={stats_csp['fp_per_min']:.3f} "
        f"| mean latency={stats_csp['mean_latency_s']:.2f}s"
    )
    print(
        f"{title}: Bandpower+LogReg win_acc={acc_bp:.3f} | blocks TP={stats_bp['tp']}/{stats_bp['n_true_blocks']} FN={stats_bp['fn']} FP={stats_bp['fp']} "
        f"| recall={stats_bp['recall']:.3f} (fists={stats_bp_fists['recall']:.3f}, feet={stats_bp_feet['recall']:.3f}) "
        f"precision={stats_bp['precision']:.3f} FP/min={stats_bp['fp_per_min']:.3f} "
        f"| mean latency={stats_bp['mean_latency_s']:.2f}s"
    )

    if do_plot:
        fig, ax = plt.subplots(figsize=(14, 5))

        gt = gt_mi.astype(int)
        diff = np.diff(np.r_[0, gt, 0])
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]
        for s0, s1 in zip(onsets, offsets):
            ax.axvspan(s0 / sfreq, s1 / sfreq, alpha=0.2)

        ax.plot(mid_times, p_csp, linewidth=1.5, label="CSP+LDA")
        ax.plot(mid_times, p_bp, linewidth=1.5, label="Bandpower+LogReg")
        ax.axhline(THRESH, linestyle="--", linewidth=1)

        ax.set_title(title + f" | win={WIN_LEN}s step={WIN_STEP}s | acc_csp={acc_csp:.3f} | acc_bp={acc_bp:.3f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("p(MI)")
        ax.legend(loc="upper right")

        ax.text(
            0.01, 0.95,
            "Shaded = true MI segments (from T1/T2 + [TMIN,TMAX])\nLines = classifier p(MI)",
            transform=ax.transAxes,
            va="top",
        )

        plt.tight_layout()
        plt.show()

    # Return dictionary with combined stats for convenience
    combined_stats = {
        "window_acc_csp": float(acc_csp),
        "window_acc_bp": float(acc_bp),
    }
    combined_stats.update({f"csp_{k}": v for k, v in stats_csp.items()})
    combined_stats.update({f"bp_{k}": v for k, v in stats_bp.items()})
    # Add per-type recalls
    combined_stats["csp_recall_fists"] = float(stats_csp_fists["recall"])
    combined_stats["csp_recall_feet"] = float(stats_csp_feet["recall"])
    combined_stats["bp_recall_fists"] = float(stats_bp_fists["recall"])
    combined_stats["bp_recall_feet"] = float(stats_bp_feet["recall"])

    return combined_stats


# -----------------------
# Proper evaluation: leave-one-run-out
# -----------------------

# Pre-load all raws
raw_by_run = {}
for r in RUNS:
    edf_path = DATA_DIR / f"{SUBJECT}R{r:02d}.edf"
    print("Loading EDF:", edf_path)
    raw_by_run[r] = load_and_preprocess_raw(edf_path)

accs = []
for test_run in RUNS:
    train_runs = [r for r in RUNS if r != test_run]

    # Build training epochs from training runs and concatenate
    train_epochs = [build_epochs_mi_rest(raw_by_run[r]) for r in train_runs]
    epochs_train = mne.concatenate_epochs(train_epochs)

    # Use SAME channel selection for training as for online simulation
    train_picks = pick_channels(epochs_train)
    X_train = epochs_train.get_data(picks=train_picks)
    y_train = (epochs_train.events[:, 2] == epochs_train.event_id["mi"]).astype(int)

    # Train a fresh model ONLY on training runs
    csp = CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("csp", csp), ("lda", lda)])
    clf.fit(X_train, y_train)

    # Train bandpower detector on the same training runs
    # Build features from epochs_train (SAME channel selection)
    roi_picks = pick_channels(epochs_train)

    X_bp = []
    for ep in X_train:
        ep_roi = ep
        feats = []
        for band in BANDS.values():
            feats.append(log_bandpower(ep_roi, sfreq=epochs_train.info["sfreq"], band=band))
        X_bp.append(np.concatenate(feats))
    X_bp = np.vstack(X_bp)

    # Z-score normalization learned on training runs
    bp_mean = X_bp.mean(axis=0)
    bp_std = X_bp.std(axis=0)
    X_bp = (X_bp - bp_mean) / (bp_std + 1e-12)

    bp_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    bp_clf.fit(X_bp, y_train)

    bp_norm = (bp_mean, bp_std)

    # Simulate online on held-out run
    do_plot = (test_run == RUNS[-1])  # plot only the last fold to keep it readable
    title = f"Held-out Task-4 run R{test_run:02d} (trained on {train_runs})"
    stats = simulate_online_on_raw(raw_by_run[test_run], clf, bp_clf, bp_norm, title=title, do_plot=do_plot)

    # Save fold models separately (so you can reuse them later)
    prefix = f"{SUBJECT}_MI_REST_TASK4_fold_heldout_R{test_run:02d}"

    csp_path = MODEL_DIR / f"{prefix}_CSP_LDA.joblib"
    bp_path = MODEL_DIR / f"{prefix}_BANDPOWER_LOGREG.joblib"
    meta_path = MODEL_DIR / f"{prefix}_META.joblib"

    # NOTE: fold saving is currently disabled (kept from your previous version).
    # Uncomment if you want per-fold models saved too.
    """
    # 1) Save CSP+LDA pipeline
    joblib.dump(clf, csp_path)

    # 2) Save Bandpower+LogReg model + normalization + channel selection metadata
    joblib.dump(
        {
            "model": bp_clf,
            "bp_norm": bp_norm,  # (mean, std)
            "bands": BANDS,
            "channel_group": CHANNEL_GROUP,
            "channels_used": [epochs_train.ch_names[i] for i in pick_channels(epochs_train)],
        },
        bp_path,
    )

    # 3) Save metadata
    meta = {
        "subject": SUBJECT,
        "held_out_run": int(test_run),
        "train_runs": [int(r) for r in train_runs],
        "freq_band": FREQ_BAND,
        "notch": NOTCH,
        "channel_group": CHANNEL_GROUP,
        "channels_used": [epochs_train.ch_names[i] for i in pick_channels(epochs_train)],
        "tmin": TMIN,
        "tmax": TMAX,
        "win_len": WIN_LEN,
        "win_step": WIN_STEP,
        "thresh": THRESH,
        "channels": raw_by_run[test_run].ch_names,
        "sfreq": float(raw_by_run[test_run].info["sfreq"]),
    }
    joblib.dump(meta, meta_path)
    """
    print("Saved fold CSP+LDA model to:", csp_path)
    print("Saved fold Bandpower+LogReg model to:", bp_path)
    print("Saved fold META to:", meta_path)

    print(
        f"Held-out R{test_run:02d} window-acc CSP: {stats['window_acc_csp']:.3f} | "
        f"window-acc BP: {stats['window_acc_bp']:.3f} | "
        f"block recall CSP: {stats['csp_recall']:.3f} | "
        f"block recall BP: {stats['bp_recall']:.3f}"
    )
    accs.append(stats)

# -----------------------
# Train + save final deployment models (trained on ALL runs)
# -----------------------
all_epochs = [build_epochs_mi_rest(raw_by_run[r]) for r in RUNS]
epochs_all = mne.concatenate_epochs(all_epochs)

# Use SAME channel selection for final model as well
final_picks = pick_channels(epochs_all)
X_all = epochs_all.get_data(picks=final_picks)
y_all = (epochs_all.events[:, 2] == epochs_all.event_id["mi"]).astype(int)

final_csp = CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)
final_lda = LinearDiscriminantAnalysis()
final_csp_lda = Pipeline([("csp", final_csp), ("lda", final_lda)])
final_csp_lda.fit(X_all, y_all)

# Bandpower features for ALL runs (SAME channel selection)
roi_picks_all = pick_channels(epochs_all)

X_bp_all = []
for ep in X_all:
    ep_roi = ep
    feats = []
    for band in BANDS.values():
        feats.append(log_bandpower(ep_roi, sfreq=epochs_all.info["sfreq"], band=band))
    X_bp_all.append(np.concatenate(feats))
X_bp_all = np.vstack(X_bp_all)

bp_mean_all = X_bp_all.mean(axis=0)
bp_std_all = X_bp_all.std(axis=0)
X_bp_all = (X_bp_all - bp_mean_all) / (bp_std_all + 1e-12)

final_bp_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
final_bp_clf.fit(X_bp_all, y_all)

final_prefix = f"{SUBJECT}_MI_REST_TASK4_FINAL_ALLRUNS"

final_csp_path = MODEL_DIR / f"{final_prefix}_CSP_LDA.joblib"
final_bp_path = MODEL_DIR / f"{final_prefix}_BANDPOWER_LOGREG.joblib"
final_meta_path = MODEL_DIR / f"{final_prefix}_META.joblib"

# 1) Save CSP+LDA pipeline
joblib.dump(final_csp_lda, final_csp_path)

# 2) Save Bandpower+LogReg model + normalization + channel selection metadata
joblib.dump(
    {
        "model": final_bp_clf,
        "bp_norm": (bp_mean_all, bp_std_all),
        "bands": BANDS,
        "channel_group": CHANNEL_GROUP,
        "channels_used": [epochs_all.ch_names[i] for i in pick_channels(epochs_all)],
    },
    final_bp_path,
)

# 3) Save metadata (helps you verify channel order / preprocessing later)
final_meta = {
    "subject": SUBJECT,
    "train_runs": [int(r) for r in RUNS],
    "freq_band": FREQ_BAND,
    "notch": NOTCH,
    "channel_group": CHANNEL_GROUP,
    "channels_used": [epochs_all.ch_names[i] for i in pick_channels(epochs_all)],
    "tmin": TMIN,
    "tmax": TMAX,
    "win_len": WIN_LEN,
    "win_step": WIN_STEP,
    "thresh": THRESH,
    "channels": epochs_all.ch_names,
    "sfreq": float(epochs_all.info["sfreq"]),
}
joblib.dump(final_meta, final_meta_path)

print("Saved FINAL CSP+LDA model to:", final_csp_path)
print("Saved FINAL Bandpower+LogReg model to:", final_bp_path)
print("Saved FINAL META to:", final_meta_path)

mean_acc_csp = float(np.mean([s["window_acc_csp"] for s in accs]))
std_acc_csp = float(np.std([s["window_acc_csp"] for s in accs]))
mean_recall_csp = float(np.mean([s["csp_recall"] for s in accs]))
std_recall_csp = float(np.std([s["csp_recall"] for s in accs]))
mean_fpmin_csp = float(np.mean([s["csp_fp_per_min"] for s in accs]))

mean_acc_bp = float(np.mean([s["window_acc_bp"] for s in accs]))
mean_recall_bp = float(np.mean([s["bp_recall"] for s in accs]))
mean_fpmin_bp = float(np.mean([s["bp_fp_per_min"] for s in accs]))

print_config()

print("Mean held-out window accuracy CSP+LDA:", mean_acc_csp, "+/-", std_acc_csp)
print("Mean held-out block recall CSP+LDA:", mean_recall_csp, "+/-", std_recall_csp)
print("Mean held-out FP/min CSP+LDA:", mean_fpmin_csp)
print("Mean held-out window accuracy Bandpower+LogReg:", mean_acc_bp)
print("Mean held-out block recall Bandpower+LogReg:", mean_recall_bp)
print("Mean held-out FP/min Bandpower+LogReg:", mean_fpmin_bp)