

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
RUNS = [
    # {
    #     "run_name": "run_28_03",
    #     "npy_file": "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/Rec_Phil_28_03_2026_RUN_1_001.npy",
    #     "csv_file": "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/tryouts/Tryouts_28_03_2026-Sheet1.csv",
    #     "phase_format": "old",
    # },

    {
        "run_name": "run_10_04",
        "npy_file": "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/ASR_10_04_Full_45Min_001.npy",
        # "npy_file": "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/EEG_10_04_Full_45Min_001.npy",
        "csv_file": "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/tryouts/EEG_Experiment_10_04_26_v1.csv",
        "phase_format": "new",
    },
]

RESULTS_DIR = Path("/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/feature_benchmark_results")

SFREQ = 250.0
CHANNEL_AXIS = 1
EEG_CHANNELS = list(range(1, 17))  # drop empty channel 0 if present

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "Cz", "P3",
    "P4", "Pz", "O1", "O2",
    "F7", "F8", "T7", "T8"
]

ROI_CHANNELS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "Pz", "O1", "O2"],
}

WINDOW_SEC = 6.0
STEP_SEC = 2.0
FEATURE_CONTEXT_RADIUS = 2

REFERENCE_MODE = "average"   # "none", "average", "custom"
CUSTOM_REFERENCE_CHANNELS = ["P3", "P4", "Pz"]
APPLY_LOG_BANDPOWER = True
LOG_EPS = 1e-12

ARTIFACT_MODE = "mask"   # "none", "mask", "suppress", "drop"
BLINK_ROI_NAME = "frontal"
BLINK_THRESHOLD_UV = 100.0
GLOBAL_ARTIFACT_THRESHOLD_UV = 150.0

BASE_EMA_SPAN = 5
SLOW_SPAN = 30
ROLL_VAR_WINDOW = 9

SLOPE_LAG = 3
PHASE_BUFFER_SEC = 10.0
PHASE_MIN_CORE_SEC = 12.0

# Exclude Phase: ["Pause"]
EXCLUDE_PHASE_KEYWORDS = []
# only exact: ["Jaw Clench Test"]
OPTIONAL_EXCLUDE_EXACT = set()

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


# For generic evaluation, use all phase pairs that have enough samples.
MIN_WINDOWS_PER_PHASE = 5
CORRELATION_METHOD = "spearman"

DEBUG_PLOTS = True
DEBUG_TOP_N_FEATURES = 12
DEBUG_ALWAYS_PLOT_FEATURES = ["alpha", "beta", "gamma", "theta"]

DEBUG_PLOT_DIR = RESULTS_DIR / "debug_plots"

CLUSTER_TOP_N_FEATURES = 80
CLUSTER_CORR_THRESHOLD = 0.80


# =========================
# LOAD / PREPROCESS
# =========================
def load_phase_table(csv_file: str, phase_format: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]

    if phase_format == "old":
        # User's earlier sheet
        phase_col = ""
        if phase_col not in df.columns:
            raise ValueError(
                f"Could not find the phase name column in old-format csv. Found columns: {df.columns.tolist()}"
            )
        df = df.rename(columns={
            phase_col: "phase",
            "Dauer": "duration_min",
            "T": "end_time_min",
        })

    elif phase_format == "new":
        required = {"Title", "Duration", "End Time"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns {sorted(missing)} in new-format csv. Found: {df.columns.tolist()}"
            )
        df = df.rename(columns={
            "Title": "phase",
            "Duration": "duration_min",
            "End Time": "end_time_min",
        })
    else:
        raise ValueError(f"Unknown phase_format: {phase_format}")

    df["phase"] = df["phase"].astype(str).str.strip()
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["end_time_min"] = pd.to_numeric(df["end_time_min"], errors="coerce")
    df["start_time_min"] = df["end_time_min"] - df["duration_min"]
    return df


def select_analysis_phases(df: pd.DataFrame) -> pd.DataFrame:
    keep_mask = np.ones(len(df), dtype=bool)

    for kw in EXCLUDE_PHASE_KEYWORDS:
        keep_mask &= ~df["phase"].str.contains(kw, case=False, na=False)

    if OPTIONAL_EXCLUDE_EXACT:
        keep_mask &= ~df["phase"].isin(OPTIONAL_EXCLUDE_EXACT)

    out = df.loc[keep_mask].copy()
    out = out.dropna(subset=["duration_min", "start_time_min", "end_time_min"])
    out = out[out["duration_min"] > 0]
    return out.reset_index(drop=True)


def prepare_data(npy_file: str, channel_axis: int = 1, eeg_channels=None) -> np.ndarray:
    data = np.load(npy_file)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    if channel_axis == 0:
        data = data.T
    elif channel_axis != 1:
        raise ValueError("CHANNEL_AXIS must be 0 or 1")

    if eeg_channels is not None:
        data = data[:, eeg_channels]

    return data.astype(np.float64)


def notch_filter(data: np.ndarray, sfreq: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    b, a = signal.iirnotch(w0=freq, Q=quality, fs=sfreq)
    return signal.filtfilt(b, a, data, axis=0)


def bandpass_filter(data: np.ndarray, sfreq: float, l_freq: float = 1.0, h_freq: float = 40.0, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def average_reference(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=1, keepdims=True)

def custom_reference(data: np.ndarray, ref_indices: list[int]) -> np.ndarray:
    if len(ref_indices) == 0:
        raise ValueError("CUSTOM_REFERENCE_CHANNELS resolved to no valid channels.")
    ref_signal = np.mean(data[:, ref_indices], axis=1, keepdims=True)
    return data - ref_signal


def resolve_roi_indices(channel_names: list[str], roi_channels: dict[str, list]) -> dict[str, list[int]]:
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    resolved = {}

    for roi_name, entries in roi_channels.items():
        idxs = []
        for entry in entries:
            if isinstance(entry, int):
                if 0 <= entry < len(channel_names):
                    idxs.append(entry)
            elif entry in name_to_idx:
                idxs.append(name_to_idx[entry])
        resolved[roi_name] = sorted(set(idxs))

    return resolved

def resolve_channel_indices(channel_names: list[str], selected_channels: list) -> list[int]:
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    idxs = []
    for entry in selected_channels:
        if isinstance(entry, int):
            if 0 <= entry < len(channel_names):
                idxs.append(entry)
        elif entry in name_to_idx:
            idxs.append(name_to_idx[entry])
    return sorted(set(idxs))


# =========================
# ARTIFACTS
# =========================
def detect_window_artifacts(window_data: np.ndarray,
                            roi_indices: dict[str, list[int]],
                            blink_roi_name: str = "frontal",
                            blink_threshold_uv: float = 120.0,
                            global_threshold_uv: float = 200.0) -> dict:
    blink_idx = roi_indices.get(blink_roi_name, [])

    if len(blink_idx) > 0:
        blink_signal = np.mean(window_data[:, blink_idx], axis=1)
        blink_ptp = float(np.ptp(blink_signal))
    else:
        blink_ptp = 0.0

    global_ptp = float(np.max(np.ptp(window_data, axis=0)))

    blink_artifact = blink_ptp > blink_threshold_uv
    global_artifact = global_ptp > global_threshold_uv
    artifact_flag = bool(blink_artifact or global_artifact)

    return {
        "blink_ptp": blink_ptp,
        "global_ptp": global_ptp,
        "blink_artifact": bool(blink_artifact),
        "global_artifact": bool(global_artifact),
        "artifact_flag": artifact_flag,
    }


def apply_artifact_mode_to_features(df: pd.DataFrame,
                                    feature_cols: list[str],
                                    mode: str = "mask") -> pd.DataFrame:
    if "artifact_flag" not in df.columns:
        return df

    if mode in {"none", "mask"}:
        return df

    if mode == "drop":
        return df.loc[~df["artifact_flag"]].reset_index(drop=True)

    if mode == "suppress":
        out = df.copy()
        artifact_idx = np.where(out["artifact_flag"].to_numpy(dtype=bool))[0]

        for i in artifact_idx:
            for col in feature_cols:
                if col not in out.columns:
                    continue

                neighbor_vals = []

                j = i - 1
                while j >= 0 and len(neighbor_vals) < 2:
                    if not bool(out.loc[j, "artifact_flag"]):
                        neighbor_vals.append(out.loc[j, col])
                    j -= 1

                j = i + 1
                while j < len(out) and len(neighbor_vals) < 4:
                    if not bool(out.loc[j, "artifact_flag"]):
                        neighbor_vals.append(out.loc[j, col])
                    j += 1

                if neighbor_vals:
                    out.loc[i, col] = float(np.median(neighbor_vals))

        return out

    raise ValueError(f"Unknown ARTIFACT_MODE: {mode}")


# =========================
# FEATURE EXTRACTION
# =========================
def compute_bandpower_window(window_data: np.ndarray, sfreq: float, bands: dict,
                             roi_indices: dict[str, list[int]] | None = None) -> dict:
    freqs, psd = signal.welch(
        window_data,
        fs=sfreq,
        nperseg=min(len(window_data), int(sfreq * 2)),
        axis=0,
        detrend="constant"
    )

    bp = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        if not np.any(idx):
            bp[band_name] = np.nan
            continue

        band_power_per_channel = np.trapezoid(psd[idx, :], freqs[idx], axis=0)
        bp[band_name] = float(np.mean(band_power_per_channel))

        if roi_indices is not None:
            if band_name == "theta":
                roi_idx = roi_indices.get("frontal", [])
                bp["theta_frontal"] = float(np.mean(band_power_per_channel[roi_idx])) if roi_idx else np.nan
            if band_name == "alpha":
                roi_idx = roi_indices.get("parietal", [])
                bp["alpha_parietal"] = float(np.mean(band_power_per_channel[roi_idx])) if roi_idx else np.nan
            if band_name == "beta":
                roi_idx = roi_indices.get("central", [])
                bp["beta_central"] = float(np.mean(band_power_per_channel[roi_idx])) if roi_idx else np.nan

    return bp


def apply_bandpower_transform(bp: dict, use_log: bool = True, eps: float = 1e-12) -> dict:
    transformed = {}
    for band_name, value in bp.items():
        if np.isnan(value):
            transformed[band_name] = np.nan
        elif use_log:
            transformed[band_name] = float(np.log10(value + eps))
        else:
            transformed[band_name] = float(value)
    return transformed


def compute_window_feature_table(data: np.ndarray, sfreq: float, roi_indices: dict[str, list[int]],
                                 artifact_mode: str = "mask") -> pd.DataFrame:
    win_samp = int(round(WINDOW_SEC * sfreq))
    step_samp = int(round(STEP_SEC * sfreq))

    rows = []
    n_samples = data.shape[0]

    for start_idx in range(0, n_samples - win_samp + 1, step_samp):
        end_idx = start_idx + win_samp
        window = data[start_idx:end_idx, :]

        artifact_info = detect_window_artifacts(
            window,
            roi_indices=roi_indices,
            blink_roi_name=BLINK_ROI_NAME,
            blink_threshold_uv=BLINK_THRESHOLD_UV,
            global_threshold_uv=GLOBAL_ARTIFACT_THRESHOLD_UV,
        )

        bp_raw = compute_bandpower_window(window, sfreq, BANDS, roi_indices=roi_indices)
        bp = apply_bandpower_transform(bp_raw, use_log=APPLY_LOG_BANDPOWER, eps=LOG_EPS)

        row = {
            "window_idx": len(rows),
            "start_sample": start_idx,
            "end_sample": end_idx,
            "start_sec": start_idx / sfreq,
            "end_sec": end_idx / sfreq,
            "center_sec": ((start_idx + end_idx) / 2) / sfreq,
        }
        row.update(artifact_info)
        row.update(bp)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Log-space ratio/balance features
    df["theta_minus_alpha"] = df["theta"] - df["alpha"]
    df["alpha_minus_beta"] = df["alpha"] - df["beta"]
    df["theta_frontal_minus_alpha_parietal"] = df["theta_frontal"] - df["alpha_parietal"]

    temporal_base_cols = [
        "theta", "alpha", "beta","gamma",
        "theta_frontal", "alpha_parietal", "beta_central",
        "theta_minus_alpha", "alpha_minus_beta", "theta_frontal_minus_alpha_parietal",
    ]

    for col in temporal_base_cols:
        if col not in df.columns:
            continue

        values = df[col].to_numpy(dtype=float)
        local_mean = []
        local_var = []
        slope = []

        for i in range(len(values)):
            start = max(0, i - FEATURE_CONTEXT_RADIUS)
            end = min(len(values), i + FEATURE_CONTEXT_RADIUS + 1)
            local_vals = values[start:end]

            local_mean.append(float(np.mean(local_vals)))
            local_var.append(float(np.var(local_vals)))

            j0 = max(0, i - SLOPE_LAG)
            if i == j0:
                slope.append(0.0)
            else:
                slope.append(float((values[i] - values[j0]) / (i - j0)))

        df[f"{col}_local_mean"] = local_mean
        df[f"{col}_local_var"] = local_var
        df[f"{col}_slope"] = slope

    feature_cols = [c for c in df.columns if c not in {
        "window_idx", "start_sample", "end_sample", "start_sec", "end_sec", "center_sec",
        "blink_ptp", "global_ptp", "blink_artifact", "global_artifact", "artifact_flag"
    }]

    df = apply_artifact_mode_to_features(df, feature_cols=feature_cols, mode=artifact_mode)
    return df


def assign_phase_to_windows(window_df: pd.DataFrame, phase_df: pd.DataFrame) -> pd.DataFrame:
    phase_names = []

    for center_sec in window_df["center_sec"]:
        assigned_phase = None
        for _, row in phase_df.iterrows():
            start_sec = row["start_time_min"] * 60.0
            end_sec = row["end_time_min"] * 60.0
            if start_sec <= center_sec < end_sec:
                assigned_phase = row["phase"]
                break
        phase_names.append(assigned_phase)

    window_df["phase"] = phase_names
    return window_df


# =========================
# GENERIC FEATURE VARIANTS
# =========================
def ema_feature(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def slow_feature(x: pd.Series, span: int) -> pd.Series:
    return ema_feature(x, span=span)


def fast_from_slow(x: pd.Series, span: int) -> pd.Series:
    return x - slow_feature(x, span=span)


def rolling_var_feature(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=1).var().fillna(0.0)


def slope_feature(x: pd.Series, lag: int) -> pd.Series:
    return x.diff(lag).fillna(0.0) / max(lag, 1)


def generate_feature_variants(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = df.copy()
    new_cols = {}

    for col in feature_names:
        if col not in out.columns:
            continue

        new_cols[f"{col}__ema_{BASE_EMA_SPAN}"] = ema_feature(out[col], span=BASE_EMA_SPAN)
        new_cols[f"{col}__slow_{SLOW_SPAN}"] = slow_feature(out[col], span=SLOW_SPAN)
        new_cols[f"{col}__fast_from_{SLOW_SPAN}"] = fast_from_slow(out[col], span=SLOW_SPAN)
        new_cols[f"{col}__var_{ROLL_VAR_WINDOW}"] = rolling_var_feature(out[col], window=ROLL_VAR_WINDOW)
        new_cols[f"{col}__slope_{SLOPE_LAG}"] = slope_feature(out[col], lag=SLOPE_LAG)

    if new_cols:
        variant_df = pd.DataFrame(new_cols, index=out.index)
        out = pd.concat([out, variant_df], axis=1)

    return out.copy()


# =========================
# EVALUATION
# =========================
def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan

    mean_diff = a.mean() - b.mean()
    pooled_std = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2.0)
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return np.nan
    return float(mean_diff / pooled_std)


def single_feature_lda_weight_and_auc(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    x = x.loc[valid]
    y = y.loc[valid]

    if y.nunique() != 2 or len(x) < 6:
        return np.nan, np.nan

    X = x.to_numpy().reshape(-1, 1)
    y_values = y.to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis()
    lda.fit(Xs, y_values)
    scores = lda.decision_function(Xs)

    classes = list(lda.classes_)
    positive_class = classes[1]
    y_bin = (y_values == positive_class).astype(int)

    try:
        auc = roc_auc_score(y_bin, scores)
    except ValueError:
        auc = np.nan

    weight = float(np.abs(lda.coef_[0][0]))
    return weight, auc


def compute_phase_stability(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for phase, phase_df in df.groupby("phase"):
        if pd.isna(phase):
            continue
        for feature in feature_cols:
            vals = pd.to_numeric(phase_df[feature], errors="coerce").dropna()
            if len(vals) < 2:
                continue
            rows.append({
                "phase": phase,
                "feature": feature,
                "n": len(vals),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "cv_abs": float(np.abs(vals.std(ddof=1) / vals.mean())) if vals.mean() != 0 else np.nan,
            })
    return pd.DataFrame(rows)



def compute_phase_internal_change(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

    if "phase" not in df.columns or "center_sec" not in df.columns:
        return pd.DataFrame(columns=[
            "phase",
            "feature",
            "core_duration_sec",
            "early_mean",
            "late_mean",
            "late_minus_early",
            "abs_late_minus_early",
            "core_std",
            "n_core_windows",
        ])

    for phase, phase_df in df.groupby("phase"):
        if pd.isna(phase):
            continue

        phase_df = phase_df.sort_values("center_sec").copy()
        if len(phase_df) < 2:
            continue

        phase_start = float(phase_df["center_sec"].min())
        phase_end = float(phase_df["center_sec"].max())
        core_start = phase_start + PHASE_BUFFER_SEC
        core_end = phase_end - PHASE_BUFFER_SEC
        core_duration_sec = core_end - core_start

        if core_duration_sec < PHASE_MIN_CORE_SEC:
            continue

        core_df = phase_df.loc[(phase_df["center_sec"] >= core_start) & (phase_df["center_sec"] <= core_end)].copy()
        if len(core_df) < 4:
            continue

        mid_sec = (core_start + core_end) / 2.0
        early_df = core_df.loc[core_df["center_sec"] < mid_sec]
        late_df = core_df.loc[core_df["center_sec"] >= mid_sec]

        if len(early_df) < 2 or len(late_df) < 2:
            continue

        for feature in feature_cols:
            if feature not in core_df.columns:
                continue

            early_vals = pd.to_numeric(early_df[feature], errors="coerce").dropna()
            late_vals = pd.to_numeric(late_df[feature], errors="coerce").dropna()
            core_vals = pd.to_numeric(core_df[feature], errors="coerce").dropna()

            if len(early_vals) < 2 or len(late_vals) < 2 or len(core_vals) < 2:
                continue

            early_mean = float(early_vals.mean())
            late_mean = float(late_vals.mean())
            late_minus_early = late_mean - early_mean

            rows.append({
                "phase": phase,
                "feature": feature,
                "core_duration_sec": float(core_duration_sec),
                "early_mean": early_mean,
                "late_mean": late_mean,
                "late_minus_early": float(late_minus_early),
                "abs_late_minus_early": float(abs(late_minus_early)),
                "core_std": float(core_vals.std(ddof=1)),
                "n_core_windows": int(len(core_vals)),
            })

    return pd.DataFrame(rows)


def compute_artifact_sensitivity(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if "artifact_flag" not in df.columns or df["artifact_flag"].nunique() < 2:
        return pd.DataFrame(columns=[
            "feature",
            "artifact_effect_size_abs",
            "n_clean_windows",
            "n_artifact_windows",
            "artifact_fraction",
        ])

    rows = []
    clean_df = df.loc[~df["artifact_flag"]]
    art_df = df.loc[df["artifact_flag"]]

    n_clean_windows = int(len(clean_df))
    n_artifact_windows = int(len(art_df))
    n_total_windows = int(len(df))
    artifact_fraction = float(n_artifact_windows / n_total_windows) if n_total_windows > 0 else np.nan

    for feature in feature_cols:
        if feature not in df.columns:
            continue
        eff = standardized_mean_difference(clean_df[feature], art_df[feature])
        rows.append({
            "feature": feature,
            "artifact_effect_size_abs": float(abs(eff)) if np.isfinite(eff) else np.nan,
            "n_clean_windows": n_clean_windows,
            "n_artifact_windows": n_artifact_windows,
            "artifact_fraction": artifact_fraction,
        })

    return pd.DataFrame(rows)


def evaluate_pairwise_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    phase_counts = df["phase"].value_counts(dropna=True)
    phases = [p for p, n in phase_counts.items() if n >= MIN_WINDOWS_PER_PHASE]

    pairs = list(combinations(sorted(phases), 2))

    for phase_a, phase_b in sorted(pairs):
        pair_df = df.loc[df["phase"].isin([phase_a, phase_b])].copy()
        if pair_df["phase"].nunique() < 2:
            continue

        for feature in feature_cols:
            if feature not in pair_df.columns:
                continue

            a_vals = pair_df.loc[pair_df["phase"] == phase_a, feature]
            b_vals = pair_df.loc[pair_df["phase"] == phase_b, feature]
            eff = standardized_mean_difference(a_vals, b_vals)
            weight, auc = single_feature_lda_weight_and_auc(pair_df[feature], pair_df["phase"])

            rows.append({
                "phase_a": phase_a,
                "phase_b": phase_b,
                "feature": feature,
                "mean_a": pd.to_numeric(a_vals, errors="coerce").mean(),
                "mean_b": pd.to_numeric(b_vals, errors="coerce").mean(),
                "mean_diff": pd.to_numeric(a_vals, errors="coerce").mean() - pd.to_numeric(b_vals, errors="coerce").mean(),
                "std_effect_size": eff,
                "abs_std_effect_size": abs(eff) if np.isfinite(eff) else np.nan,
                "single_feature_lda_weight_abs": weight,
                "single_feature_auc": auc,
                "n_a": int(pd.to_numeric(a_vals, errors="coerce").notna().sum()),
                "n_b": int(pd.to_numeric(b_vals, errors="coerce").notna().sum()),
            })

    return pd.DataFrame(rows)


def summarize_feature_rankings(pairwise_df: pd.DataFrame,
                               stability_df: pd.DataFrame,
                               phase_change_df: pd.DataFrame,
                               artifact_df: pd.DataFrame) -> pd.DataFrame:
    summary = pairwise_df.groupby("feature").agg(
        mean_abs_effect_size=("abs_std_effect_size", "mean"),
        median_abs_effect_size=("abs_std_effect_size", "median"),
        mean_abs_lda_weight=("single_feature_lda_weight_abs", "mean"),
        mean_auc=("single_feature_auc", "mean"),
        n_pairs=("feature", "count"),
    ).reset_index()

    if not stability_df.empty:
        stability_summary = stability_df.groupby("feature").agg(
            mean_phase_std=("std", "mean"),
            mean_phase_cv_abs=("cv_abs", "mean"),
        ).reset_index()
        summary = summary.merge(stability_summary, on="feature", how="left")

    if not phase_change_df.empty:
        phase_change_summary = phase_change_df.groupby("feature").agg(
            mean_abs_phase_change=("abs_late_minus_early", "mean"),
            median_abs_phase_change=("abs_late_minus_early", "median"),
            mean_core_std=("core_std", "mean"),
        ).reset_index()
        summary = summary.merge(phase_change_summary, on="feature", how="left")

    if not artifact_df.empty:
        artifact_summary = artifact_df[[
            "feature",
            "artifact_effect_size_abs",
            "n_clean_windows",
            "n_artifact_windows",
            "artifact_fraction",
        ]].drop_duplicates(subset=["feature"])
        summary = summary.merge(artifact_summary, on="feature", how="left")

    summary = summary.sort_values(
        by=["mean_abs_effect_size", "mean_abs_lda_weight", "mean_auc"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return summary


def cluster_ranked_features(full_df: pd.DataFrame,
                            ranking_df: pd.DataFrame,
                            feature_cols: list[str],
                            top_n: int = 80,
                            corr_threshold: float = 0.90,
                            method: str = "spearman") -> tuple[pd.DataFrame, pd.DataFrame]:
    if ranking_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ranked_features = [f for f in ranking_df["feature"].head(top_n).tolist() if f in feature_cols and f in full_df.columns]
    if len(ranked_features) == 0:
        return pd.DataFrame(), pd.DataFrame()

    if len(ranked_features) == 1:
        only_feature = ranked_features[0]
        assignments = pd.DataFrame([{
            "feature": only_feature,
            "cluster_id": 1,
        }])
        representatives = ranking_df.loc[ranking_df["feature"] == only_feature].copy()
        representatives.insert(1, "cluster_id", 1)
        return assignments, representatives

    corr_df = full_df[ranked_features].corr(method=method)
    corr_df = corr_df.fillna(0.0)

    distance_df = 1.0 - corr_df.abs()
    np.fill_diagonal(distance_df.values, 0.0)

    condensed = squareform(distance_df.values, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")

    assignments = pd.DataFrame({
        "feature": ranked_features,
        "cluster_id": cluster_ids,
    })

    cluster_sizes = assignments.groupby("cluster_id").size().rename("cluster_size").reset_index()
    assignments = assignments.merge(cluster_sizes, on="cluster_id", how="left")

    assignments = assignments.merge(
        ranking_df.reset_index().rename(columns={"index": "ranking_position"}),
        on="feature",
        how="left",
    )

    representatives = assignments.sort_values(
        by=["cluster_id", "mean_abs_effect_size", "mean_abs_lda_weight", "mean_auc", "ranking_position"],
        ascending=[True, False, False, False, True],
    ).groupby("cluster_id", as_index=False).first()

    representatives = representatives.sort_values("ranking_position").reset_index(drop=True)
    return assignments, representatives


# =========================
# FEATURE CLUSTER SUMMARY
# =========================

def summarize_feature_clusters(cluster_assignments_df: pd.DataFrame,
                               cluster_representatives_df: pd.DataFrame) -> pd.DataFrame:
    if cluster_assignments_df.empty:
        return pd.DataFrame(columns=[
            "cluster_id",
            "cluster_size",
            "representative_feature",
            "member_features",
        ])

    member_summary = cluster_assignments_df.groupby("cluster_id").agg(
        cluster_size=("feature", "size"),
        member_features=("feature", lambda x: "; ".join(sorted(map(str, x)))),
    ).reset_index()

    rep_summary = cluster_representatives_df[["cluster_id", "feature"]].rename(
        columns={"feature": "representative_feature"}
    )

    cluster_summary_df = member_summary.merge(rep_summary, on="cluster_id", how="left")
    cluster_summary_df = cluster_summary_df[
        ["cluster_id", "cluster_size", "representative_feature", "member_features"]
    ].sort_values("cluster_id").reset_index(drop=True)

    return cluster_summary_df


def plot_feature_debug_timeseries(df: pd.DataFrame, feature_name: str, out_dir: Path):
    if feature_name not in df.columns:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    phase_names = [p for p in df["phase"].dropna().unique()]
    cmap = plt.get_cmap("tab20")
    phase_to_color = {phase: cmap(i % 20) for i, phase in enumerate(phase_names)}

    plt.figure(figsize=(14, 6))

    phase_segments = []
    current_phase = None
    start_sec = None
    prev_sec = None

    for _, row in df.iterrows():
        phase = row["phase"]
        center_sec = row["center_sec"]
        if phase != current_phase:
            if current_phase is not None and start_sec is not None and prev_sec is not None:
                phase_segments.append((current_phase, start_sec, prev_sec))
            current_phase = phase
            start_sec = center_sec
        prev_sec = center_sec

    if current_phase is not None and start_sec is not None and prev_sec is not None:
        phase_segments.append((current_phase, start_sec, prev_sec))

    for phase, seg_start, seg_end in phase_segments:
        plt.axvspan(seg_start, seg_end, color=phase_to_color.get(phase, "gray"), alpha=0.06)

    if "artifact_flag" in df.columns:
        for _, row in df.loc[df["artifact_flag"]].iterrows():
            plt.axvspan(row["start_sec"], row["end_sec"], color="red", alpha=0.05)

    for run_name, run_df in df.groupby("run_name"):
        plt.plot(run_df["center_sec"], run_df[feature_name], label=run_name)

    y_min, y_max = plt.ylim()
    y_text = y_max - 0.03 * (y_max - y_min)

    for phase, seg_start, seg_end in phase_segments:
        plt.text(
            (seg_start + seg_end) / 2,
            y_text,
            str(phase),
            ha="center",
            va="top",
            fontsize=8,
            rotation=45,
            alpha=0.8,
        )

    plt.xlabel("Time (s)")
    plt.ylabel(feature_name)
    plt.title(f"Debug feature trace: {feature_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = feature_name.replace("/", "_")
    plt.savefig(out_dir / f"debug_timeseries_{safe_name}.png", dpi=200)
    plt.close()


def make_debug_plots(full_df: pd.DataFrame,
                     ranking_df: pd.DataFrame,
                     out_dir: Path,
                     top_n: int = 12,
                     always_plot_features: list[str] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    top_features = []
    if not ranking_df.empty:
        top_features = ranking_df["feature"].head(top_n).tolist()

    always_plot_features = always_plot_features or []

    features_to_plot = []
    for feature_name in top_features + always_plot_features:
        if feature_name in full_df.columns and feature_name not in features_to_plot:
            features_to_plot.append(feature_name)

    for feature_name in features_to_plot:
        plot_feature_debug_timeseries(full_df, feature_name, out_dir)


# =========================
# RUN PIPELINE
# =========================
def process_single_run(run_cfg: dict) -> pd.DataFrame:
    phase_df = load_phase_table(run_cfg["csv_file"], run_cfg["phase_format"])
    phase_df = select_analysis_phases(phase_df)

    data = prepare_data(run_cfg["npy_file"], channel_axis=CHANNEL_AXIS, eeg_channels=EEG_CHANNELS)
    if len(CHANNEL_NAMES) != data.shape[1]:
        raise ValueError(
            f"CHANNEL_NAMES length ({len(CHANNEL_NAMES)}) does not match number of EEG channels ({data.shape[1]})."
        )

    roi_indices = resolve_roi_indices(CHANNEL_NAMES, ROI_CHANNELS)

    data = notch_filter(data, SFREQ, freq=50.0, quality=30.0)
    data = bandpass_filter(data, SFREQ, l_freq=1.0, h_freq=40.0, order=4)
    if REFERENCE_MODE == "average":
        data = average_reference(data)
    elif REFERENCE_MODE == "custom":
        ref_indices = resolve_channel_indices(CHANNEL_NAMES, CUSTOM_REFERENCE_CHANNELS)
        data = custom_reference(data, ref_indices)
    elif REFERENCE_MODE == "none":
        pass
    else:
        raise ValueError(f"Unknown REFERENCE_MODE: {REFERENCE_MODE}")

    feature_df = compute_window_feature_table(
        data,
        SFREQ,
        roi_indices=roi_indices,
        artifact_mode=ARTIFACT_MODE,
    )
    feature_df = assign_phase_to_windows(feature_df, phase_df)
    feature_df["run_name"] = run_cfg["run_name"]

    base_feature_cols = [
        c for c in feature_df.columns
        if c not in {
            "run_name", "phase", "window_idx", "start_sample", "end_sample", "start_sec", "end_sec", "center_sec",
            "blink_ptp", "global_ptp", "blink_artifact", "global_artifact", "artifact_flag"
        }
    ]

    feature_df = generate_feature_variants(feature_df, base_feature_cols)
    return feature_df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_run_dfs = []
    for run_cfg in RUNS:
        print(f"Processing {run_cfg['run_name']} ...")
        run_df = process_single_run(run_cfg)
        all_run_dfs.append(run_df)

        run_path = RESULTS_DIR / f"{run_cfg['run_name']}_window_features_full.csv"
        run_df.to_csv(run_path, index=False)
        print(f"Saved {run_path}")

    full_df = pd.concat(all_run_dfs, ignore_index=True)
    full_df.to_csv(RESULTS_DIR / "all_runs_window_features_full.csv", index=False)

    meta_cols = {
        "run_name", "phase", "window_idx", "start_sample", "end_sample", "start_sec", "end_sec", "center_sec",
        "blink_ptp", "global_ptp", "blink_artifact", "global_artifact", "artifact_flag"
    }
    feature_cols = [c for c in full_df.columns if c not in meta_cols]

    pairwise_df = evaluate_pairwise_features(full_df, feature_cols)
    pairwise_df.to_csv(RESULTS_DIR / "feature_pairwise_scores.csv", index=False)

    stability_df = compute_phase_stability(full_df, feature_cols)
    stability_df.to_csv(RESULTS_DIR / "feature_phase_stability.csv", index=False)

    phase_change_df = compute_phase_internal_change(full_df, feature_cols)
    phase_change_df.to_csv(RESULTS_DIR / "feature_phase_internal_change.csv", index=False)

    artifact_df = compute_artifact_sensitivity(full_df, feature_cols)
    artifact_df.to_csv(RESULTS_DIR / "feature_artifact_sensitivity.csv", index=False)

    artifact_counts = pd.DataFrame([
        {
            "run_name": run_name,
            "n_total_windows": int(len(run_part)),
            "n_artifact_windows": int(run_part["artifact_flag"].sum()) if "artifact_flag" in run_part.columns else 0,
            "n_clean_windows": int((~run_part["artifact_flag"]).sum()) if "artifact_flag" in run_part.columns else int(len(run_part)),
            "artifact_fraction": float(run_part["artifact_flag"].mean()) if "artifact_flag" in run_part.columns and len(run_part) > 0 else np.nan,
        }
        for run_name, run_part in full_df.groupby("run_name")
    ])
    artifact_counts.to_csv(RESULTS_DIR / "artifact_window_counts.csv", index=False)

    ranking_df = summarize_feature_rankings(pairwise_df, stability_df, phase_change_df, artifact_df)
    ranking_df.to_csv(RESULTS_DIR / "feature_ranking_summary.csv", index=False)

    cluster_assignments_df, cluster_representatives_df = cluster_ranked_features(
        full_df,
        ranking_df,
        feature_cols,
        top_n=CLUSTER_TOP_N_FEATURES,
        corr_threshold=CLUSTER_CORR_THRESHOLD,
        method=CORRELATION_METHOD,
    )
    cluster_assignments_df.to_csv(RESULTS_DIR / "feature_cluster_assignments.csv", index=False)
    cluster_representatives_df.to_csv(RESULTS_DIR / "feature_cluster_representatives.csv", index=False)

    cluster_summary_df = summarize_feature_clusters(
        cluster_assignments_df,
        cluster_representatives_df,
    )
    cluster_summary_df.to_csv(RESULTS_DIR / "feature_cluster_summary.csv", index=False)

    if DEBUG_PLOTS:
        make_debug_plots(
            full_df,
            ranking_df,
            DEBUG_PLOT_DIR,
            top_n=DEBUG_TOP_N_FEATURES,
            always_plot_features=DEBUG_ALWAYS_PLOT_FEATURES,
        )

    corr_df = full_df[feature_cols].corr(method=CORRELATION_METHOD)
    corr_df.to_csv(RESULTS_DIR / "feature_correlation_matrix.csv")

    #rint("\nTop 20 features:")
    #print(ranking_df.head(20).to_string(index=False))
    print(f"\nSaved all outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()