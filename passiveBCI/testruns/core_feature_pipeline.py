import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CORE EEG FEATURE PIPELINE
# Combines:
# - simple interpretable feature extraction from plot_control_axes
# - quantitative feature relevance estimation from feature_benchmark_pipeline
#
# Main idea:
# EEG -> bandpower -> temporal variance -> EMA fast/slow streams -> optional z-scoring
#      -> optional formula-based axes -> plots + relevance tables
# ============================================================


# =========================
# CONFIG
# =========================
RUN_NAME = "run_26_04_B"

if RUN_NAME == "run_10_04":
    NPY_FILE = "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/ASR_10_04_Full_45Min_001.npy"
    CSV_FILE = "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/tryouts/EEG_Experiment_10_04_26_v1.csv"
elif RUN_NAME == "run_26_04":
    NPY_FILE = "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/Rec_26_04_Phil_A/ASR_26_04_Phil_A_001.npy"
    CSV_FILE = "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/Rec_26_04_Phil_A/EEG_Experiment_26_04_26_v2.csv"
elif RUN_NAME == "run_26_04_B":
    NPY_FILE = "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/Rec_26_04_Phil_B/ASR_26_04_Phil_B_001.npy"
    CSV_FILE = "/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/data/Rec_26_04_Phil_A/EEG_Experiment_26_04_26_v2.csv"
else:
    raise ValueError(f"Unknown RUN_NAME: {RUN_NAME}")

RESULTS_DIR = Path(
    f"/Users/jakobnieder/Documents/BurgtheaterEEG/EEGTheater/passiveBCI/testruns/core_feature_results_{RUN_NAME}"
)

SFREQ = 250.0
CHANNEL_AXIS = 1
EEG_CHANNELS = list(range(1, 17))


CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "P3", "P4",
    "O1", "O2", "T7", "T8",
    "Fz", "Cpz", "Tp9", "Tp10",
]


# ROI feature definitions for bandpower and variance features.
ROI_FEATURES = {
    "theta_frontal": {
        "band": "theta",
        "channels": ["Fp1", "Fp2", "F3", "F4", "Fz"],
    },
    "alpha_parietal": {
        "band": "alpha",
        "channels": ["P3", "P4", "O1", "O2"],
    },
    "beta_central": {
        "band": "beta",
        "channels": ["C3", "C4", "Cpz"],
    },
}

# Channel-level features needed for frontal alpha asymmetry.
# These are not meant to explode the feature space; they are only kept for this specific hypothesis.
CHANNEL_BAND_FEATURES = {
    "alpha_F3": {
        "band": "alpha",
        "channel": "F3",
    },
    "alpha_F4": {
        "band": "alpha",
        "channel": "F4",
    },
}


ASYMMETRY_FEATURES = ["frontal_alpha_asymmetry"]

# Ratio / balance features.
# Because bandpowers are log-transformed, subtraction is equivalent to a log-ratio:
# alpha_theta_ratio = log(alpha power) - log(theta power) = log(alpha/theta).
RATIO_FEATURES = {
    "alpha_theta_ratio": {
        "numerator": "alpha",
        "denominator": "theta",
    },
    "alpha_parietal_theta_frontal_ratio": {
        "numerator": "alpha_parietal",
        "denominator": "theta_frontal",
    },
}

REFERENCE_MODE = "average"  # "none", "average", "custom"
CUSTOM_REFERENCE_CHANNELS = ["Tp9", "Tp10"]

WINDOW_SEC = 6.0
STEP_SEC = 2.0

# Core bands only. Keep this list deliberately small.
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

APPLY_LOG_BANDPOWER = True
LOG_EPS = 1e-12

# Artifact detection only; default does not alter features.
ARTIFACT_MODE = "mask"  # "none", "mask", "suppress", "drop"
BLINK_ROI_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "Fz"]
BLINK_THRESHOLD_UV = 100.0
GLOBAL_ARTIFACT_THRESHOLD_UV = 200.0

# Temporal feature variants.
# EMA spans are in windows. With STEP_SEC = 2:
# 5 windows ~ 10 s, 30 windows ~ 60 s.
EMA_FAST_SPAN = 30
EMA_SLOW_SPAN = 90
VAR_EMA_SPAN = 9

# Offline diagnostic normalization.
# This is NOT online-ready. It is only useful for plotting / debugging.
ADD_GLOBAL_ZSCORE_COLUMNS = True

# Phase analysis.
PHASE_BUFFER_SEC = 10.0
PHASE_MIN_CORE_SEC = 12.0
MIN_WINDOWS_PER_PHASE = 5

# Plotting.
PLOT_FEATURES = [
    '''
    # global bandpower features
    "theta_z", "alpha_z", "beta_z", "gamma_z",

    # ROI bandpower features for comparison against global features
    "theta_frontal_z", "alpha_parietal_z", "beta_central_z",
    "alpha_theta_ratio_z", "alpha_parietal_theta_frontal_ratio_z",
    "frontal_alpha_asymmetry_z", "frontal_alpha_asymmetry_ema_slow_z",

    # global variance features
    "theta_var_z", "alpha_var_z", "beta_var_z", "gamma_var_z",

    # ROI variance features
    "theta_frontal_var_z", "alpha_parietal_var_z", "beta_central_var_z",
    '''
    "alpha_ema_slow_z", "alpha_parietal_ema_slow_z"
]
PLOT_AXES = True
PLOT_BASE_FEATURES = True


# =========================
# LOADING / PREPROCESSING
# =========================
def load_phase_table(csv_file: str) -> pd.DataFrame:
    wanted_cols = {"Type", "Title", "Duration", "End Time"}
    df = pd.read_csv(
        csv_file,
        usecols=lambda c: str(c).strip() in wanted_cols,
        engine="python",
    )
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Title", "Duration", "End Time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_file}. Found: {df.columns.tolist()}")

    df = df.rename(columns={
        "Title": "phase",
        "Duration": "duration_min",
        "End Time": "end_time_min",
    })

    df["phase"] = df["phase"].astype(str).str.strip()
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["end_time_min"] = pd.to_numeric(df["end_time_min"], errors="coerce")
    df["start_time_min"] = df["end_time_min"] - df["duration_min"]

    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    df = df.dropna(subset=["duration_min", "start_time_min", "end_time_min"])
    df = df[df["duration_min"] > 0].reset_index(drop=True)
    return df


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


def custom_reference(data: np.ndarray, ref_indices: list[int]) -> np.ndarray:
    if len(ref_indices) == 0:
        raise ValueError("CUSTOM_REFERENCE_CHANNELS resolved to no valid channels.")
    ref_signal = np.mean(data[:, ref_indices], axis=1, keepdims=True)
    return data - ref_signal


def apply_reference(data: np.ndarray) -> np.ndarray:
    if REFERENCE_MODE == "average":
        return average_reference(data)
    if REFERENCE_MODE == "custom":
        ref_indices = resolve_channel_indices(CHANNEL_NAMES, CUSTOM_REFERENCE_CHANNELS)
        return custom_reference(data, ref_indices)
    if REFERENCE_MODE == "none":
        return data
    raise ValueError(f"Unknown REFERENCE_MODE: {REFERENCE_MODE}")


# =========================
# ARTIFACTS
# =========================
def detect_window_artifacts(window_data: np.ndarray) -> dict:
    blink_idx = resolve_channel_indices(CHANNEL_NAMES, BLINK_ROI_CHANNELS)

    if len(blink_idx) > 0:
        blink_signal = np.mean(window_data[:, blink_idx], axis=1)
        blink_ptp = float(np.ptp(blink_signal))
    else:
        blink_ptp = 0.0

    global_ptp = float(np.max(np.ptp(window_data, axis=0)))

    blink_artifact = blink_ptp > BLINK_THRESHOLD_UV
    global_artifact = global_ptp > GLOBAL_ARTIFACT_THRESHOLD_UV
    artifact_flag = bool(blink_artifact or global_artifact)

    return {
        "blink_ptp": blink_ptp,
        "global_ptp": global_ptp,
        "blink_artifact": bool(blink_artifact),
        "global_artifact": bool(global_artifact),
        "artifact_flag": artifact_flag,
    }


def apply_artifact_mode_to_features(df: pd.DataFrame, feature_cols: list[str], mode: str) -> pd.DataFrame:
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

                left_vals = []
                right_vals = []

                j = i - 1
                while j >= 0 and len(left_vals) < 2:
                    if not bool(out.loc[j, "artifact_flag"]):
                        left_vals.append(out.loc[j, col])
                    j -= 1

                j = i + 1
                while j < len(out) and len(right_vals) < 2:
                    if not bool(out.loc[j, "artifact_flag"]):
                        right_vals.append(out.loc[j, col])
                    j += 1

                neighbor_vals = left_vals + right_vals
                if neighbor_vals:
                    out.loc[i, col] = float(np.median(neighbor_vals))

        return out

    raise ValueError(f"Unknown ARTIFACT_MODE: {mode}")


# =========================
# FEATURE EXTRACTION
# =========================
def compute_bandpower_window(window_data: np.ndarray, sfreq: float, bands: dict) -> dict:
    freqs, psd = signal.welch(
        window_data,
        fs=sfreq,
        nperseg=min(len(window_data), int(sfreq * 2)),
        axis=0,
        detrend="constant",
    )

    bp = {}
    band_power_cache = {}

    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        if not np.any(idx):
            bp[band_name] = np.nan
            band_power_cache[band_name] = None
            continue

        band_power_per_channel = np.trapezoid(psd[idx, :], freqs[idx], axis=0)
        band_power_cache[band_name] = band_power_per_channel

        # Global bandpower: mean across all EEG channels.
        # These global features are intentionally kept so they can be compared to ROI features.
        bp[band_name] = float(np.mean(band_power_per_channel))

    # ROI bandpower features: frontal theta, parietal alpha, central beta.
    for roi_feature_name, cfg in ROI_FEATURES.items():
        band_name = cfg["band"]
        channels = cfg["channels"]
        band_power_per_channel = band_power_cache.get(band_name)

        if band_power_per_channel is None:
            bp[roi_feature_name] = np.nan
            continue

        roi_indices = resolve_channel_indices(CHANNEL_NAMES, channels)
        if len(roi_indices) == 0:
            bp[roi_feature_name] = np.nan
            continue

        bp[roi_feature_name] = float(np.mean(band_power_per_channel[roi_indices]))

    # Minimal channel-level alpha features for frontal alpha asymmetry.
    for channel_feature_name, cfg in CHANNEL_BAND_FEATURES.items():
        band_name = cfg["band"]
        channel_name = cfg["channel"]
        band_power_per_channel = band_power_cache.get(band_name)

        if band_power_per_channel is None:
            bp[channel_feature_name] = np.nan
            continue

        channel_indices = resolve_channel_indices(CHANNEL_NAMES, [channel_name])
        if len(channel_indices) == 0:
            bp[channel_feature_name] = np.nan
            continue

        bp[channel_feature_name] = float(band_power_per_channel[channel_indices[0]])

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


def compute_window_feature_table(data: np.ndarray) -> pd.DataFrame:
    win_samp = int(round(WINDOW_SEC * SFREQ))
    step_samp = int(round(STEP_SEC * SFREQ))
    n_samples = data.shape[0]

    rows = []

    for start_idx in range(0, n_samples - win_samp + 1, step_samp):
        end_idx = start_idx + win_samp
        window = data[start_idx:end_idx, :]

        artifact_info = detect_window_artifacts(window)

        bp_raw = compute_bandpower_window(window, SFREQ, BANDS)
        bp = apply_bandpower_transform(bp_raw, use_log=APPLY_LOG_BANDPOWER, eps=LOG_EPS)

        row = {
            "window_idx": len(rows),
            "start_sample": start_idx,
            "end_sample": end_idx,
            "start_sec": start_idx / SFREQ,
            "end_sec": end_idx / SFREQ,
            "center_sec": ((start_idx + end_idx) / 2) / SFREQ,
        }
        row.update(artifact_info)
        row.update(bp)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Frontal alpha asymmetry / FAA proxy.
    # Because bandpower values are log-transformed later, this difference is a log-ratio:
    # positive values mean alpha_F4 > alpha_F3.
    # Since alpha is commonly interpreted as inverse cortical activation,
    # this is often read as relatively stronger left-frontal activation.
    if "alpha_F4" in df.columns and "alpha_F3" in df.columns:
        df["frontal_alpha_asymmetry"] = df["alpha_F4"] - df["alpha_F3"]

    # Log-ratio / balance features.
    # These are calculated after log-transforming the bandpowers, so subtraction represents log(numerator/denominator).
    for ratio_name, cfg in RATIO_FEATURES.items():
        numerator = cfg["numerator"]
        denominator = cfg["denominator"]
        if numerator in df.columns and denominator in df.columns:
            df[ratio_name] = df[numerator] - df[denominator]

    core_feature_cols = list(BANDS.keys()) + list(ROI_FEATURES.keys()) + ASYMMETRY_FEATURES + list(RATIO_FEATURES.keys())
    df = add_variance_features(df, core_feature_cols, span=VAR_EMA_SPAN)
    df = apply_artifact_mode_to_features(
        df,
        feature_cols=core_feature_cols + [f"{feature}_var" for feature in core_feature_cols],
        mode=ARTIFACT_MODE,
    )

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
# TEMPORAL TRANSFORMS
# =========================
def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def add_variance_features(df: pd.DataFrame, feature_cols: list[str], span: int) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        mean_fast = ema(out[col], span=span)
        out[f"{col}_var"] = ema((out[col] - mean_fast) ** 2, span=span)
    return out


def add_ema_variants(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    new_cols = {}

    for col in feature_cols:
        if col not in out.columns:
            continue

        fast_col = f"{col}_ema_fast"
        slow_col = f"{col}_ema_slow"
        residual_col = f"{col}_fast_minus_slow"

        new_cols[fast_col] = ema(out[col], span=EMA_FAST_SPAN)
        new_cols[slow_col] = ema(out[col], span=EMA_SLOW_SPAN)
        new_cols[residual_col] = new_cols[fast_col] - new_cols[slow_col]

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out.copy()


def add_global_zscore_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    new_cols = {}

    for col in feature_cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        mean = values.mean()
        std = values.std()
        if std > 0:
            new_cols[f"{col}_z"] = (values - mean) / std
        else:
            new_cols[f"{col}_z"] = np.nan

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out.copy()


def create_feature_variants(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base_features = list(BANDS.keys()) + list(ROI_FEATURES.keys()) + ASYMMETRY_FEATURES + list(RATIO_FEATURES.keys())
    core_features = base_features + [f"{feature}_var" for feature in base_features]

    df = add_ema_variants(df, core_features)

    created_features = []
    for col in df.columns:
        if col in core_features:
            created_features.append(col)
        elif any(col.startswith(f"{base}_") for base in core_features):
            created_features.append(col)

    if ADD_GLOBAL_ZSCORE_COLUMNS:
        df = add_global_zscore_columns(df, created_features)
        created_features += [f"{col}_z" for col in created_features if f"{col}_z" in df.columns]

    return df, created_features


# =========================
# AXIS FORMULAS
# =========================
def compute_axes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Define your interpretable control axes here.

    IMPORTANT:
    - Prefer *_z columns for offline debugging because they are comparable on the same scale.
    - For live use, replace global z-scoring with baseline/online normalization.
    - Axis names are defined here and automatically plotted.
    """
    axis_cols = []
    
    # Example 1: alpha amplitude / internalness-like axis.
    # Prefer parietal alpha when available, but keep global alpha as fallback.
    if "alpha_parietal_ema_slow_z" in df.columns:
        df["axis_alpha_slow"] = df["alpha_parietal_ema_slow_z"]
        axis_cols.append("axis_alpha_slow")
    elif "alpha_ema_slow_z" in df.columns:
        df["axis_alpha_slow"] = df["alpha_ema_slow_z"]
        axis_cols.append("axis_alpha_slow")

    # Example 2: alpha variability / volatility-like axis
    if "alpha_var_ema_fast_z" in df.columns:
        df["axis_alpha_variability"] = df["alpha_var_ema_fast_z"]
        axis_cols.append("axis_alpha_variability")

    # Example 3: cautious theta-vs-alpha balance.
    # Prefer ROI versions when available, but keep global features as fallback.
    if "theta_frontal_ema_slow_z" in df.columns and "alpha_parietal_ema_slow_z" in df.columns:
        df["axis_theta_alpha_balance"] = df["theta_frontal_ema_slow_z"] - df["alpha_parietal_ema_slow_z"]
        axis_cols.append("axis_theta_alpha_balance")
    elif "theta_ema_slow_z" in df.columns and "alpha_ema_slow_z" in df.columns:
        df["axis_theta_alpha_balance"] = df["theta_ema_slow_z"] - df["alpha_ema_slow_z"]
        axis_cols.append("axis_theta_alpha_balance")
    

    '''    
    df['axis_alpha'] = df["alpha_z"]
    axis_cols.append["axis_alpha"]
    '''

    # Alpha-theta ratio axis.
    # Positive values mean relatively stronger alpha than theta.
    if "alpha_parietal_theta_frontal_ratio_ema_slow_z" in df.columns:
        df["axis_alpha_theta_ratio"] = df["alpha_parietal_theta_frontal_ratio_ema_slow_z"]
        axis_cols.append("axis_alpha_theta_ratio")
    elif "alpha_theta_ratio_ema_slow_z" in df.columns:
        df["axis_alpha_theta_ratio"] = df["alpha_theta_ratio_ema_slow_z"]
        axis_cols.append("axis_alpha_theta_ratio")

    # Experimental valence / approach-withdrawal proxy based on frontal alpha asymmetry.
    # Treat this as exploratory, not as a robust direct valence readout.
    if "frontal_alpha_asymmetry_ema_slow_z" in df.columns:
        df["axis_valence_faa"] = df["frontal_alpha_asymmetry_ema_slow_z"]
        axis_cols.append("axis_valence_faa")

    return df, axis_cols


# =========================
# RELEVANCE ESTIMATION
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


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    x = x.loc[valid]
    y = y.loc[valid]

    if y.nunique() != 2 or len(x) < 6:
        return np.nan

    X = x.to_numpy().reshape(-1, 1)
    y_values = y.to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LinearDiscriminantAnalysis()
    clf.fit(Xs, y_values)
    scores = clf.decision_function(Xs)

    classes = list(clf.classes_)
    y_bin = (y_values == classes[1]).astype(int)

    try:
        return float(roc_auc_score(y_bin, scores))
    except ValueError:
        return np.nan


def compute_feature_relevance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    phase_counts = df["phase"].value_counts(dropna=True)
    phases = sorted([p for p, n in phase_counts.items() if n >= MIN_WINDOWS_PER_PHASE])

    rows = []

    for phase_a, phase_b in combinations(phases, 2):
        pair_df = df.loc[df["phase"].isin([phase_a, phase_b])].copy()
        if pair_df["phase"].nunique() < 2:
            continue

        for feature in feature_cols:
            if feature not in pair_df.columns:
                continue

            a_vals = pair_df.loc[pair_df["phase"] == phase_a, feature]
            b_vals = pair_df.loc[pair_df["phase"] == phase_b, feature]

            effect = standardized_mean_difference(a_vals, b_vals)
            auc = single_feature_auc(pair_df[feature], pair_df["phase"])

            rows.append({
                "phase_a": phase_a,
                "phase_b": phase_b,
                "feature": feature,
                "mean_a": pd.to_numeric(a_vals, errors="coerce").mean(),
                "mean_b": pd.to_numeric(b_vals, errors="coerce").mean(),
                "mean_diff": pd.to_numeric(a_vals, errors="coerce").mean() - pd.to_numeric(b_vals, errors="coerce").mean(),
                "std_effect_size": effect,
                "abs_std_effect_size": abs(effect) if np.isfinite(effect) else np.nan,
                "single_feature_auc": auc,
                "n_a": int(pd.to_numeric(a_vals, errors="coerce").notna().sum()),
                "n_b": int(pd.to_numeric(b_vals, errors="coerce").notna().sum()),
            })

    return pd.DataFrame(rows)


def summarize_relevance(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()

    summary = pairwise_df.groupby("feature").agg(
        mean_abs_effect_size=("abs_std_effect_size", "mean"),
        median_abs_effect_size=("abs_std_effect_size", "median"),
        mean_auc=("single_feature_auc", "mean"),
        n_pairs=("feature", "count"),
    ).reset_index()

    return summary.sort_values(
        by=["mean_abs_effect_size", "mean_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)


def compute_phase_internal_change(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

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

            late_minus_early = float(late_vals.mean() - early_vals.mean())

            rows.append({
                "phase": phase,
                "feature": feature,
                "core_duration_sec": float(core_duration_sec),
                "early_mean": float(early_vals.mean()),
                "late_mean": float(late_vals.mean()),
                "late_minus_early": late_minus_early,
                "abs_late_minus_early": abs(late_minus_early),
                "core_std": float(core_vals.std(ddof=1)),
                "n_core_windows": int(len(core_vals)),
            })

    return pd.DataFrame(rows)


def compute_artifact_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "artifact_flag" not in df.columns:
        return pd.DataFrame([{
            "n_total_windows": len(df),
            "n_artifact_windows": 0,
            "n_clean_windows": len(df),
            "artifact_fraction": np.nan,
        }])

    n_total = int(len(df))
    n_art = int(df["artifact_flag"].sum())
    return pd.DataFrame([{
        "n_total_windows": n_total,
        "n_artifact_windows": n_art,
        "n_clean_windows": n_total - n_art,
        "artifact_fraction": float(n_art / n_total) if n_total > 0 else np.nan,
    }])


# =========================
# PLOTTING
# =========================
def plot_time_series(df: pd.DataFrame,
                     phase_df: pd.DataFrame,
                     cols_to_plot: list[str],
                     title: str,
                     output_file: Path,
                     ylabel: str = "value"):
    phase_names = [p for p in phase_df["phase"].dropna().unique()]
    cmap = plt.get_cmap("tab20")
    phase_to_color = {phase: cmap(i % 20) for i, phase in enumerate(phase_names)}

    plt.figure(figsize=(15, 7))

    for _, row in phase_df.iterrows():
        phase = row["phase"]
        start_sec = row["start_time_min"] * 60.0
        end_sec = row["end_time_min"] * 60.0
        plt.axvspan(start_sec, end_sec, color=phase_to_color[phase], alpha=0.06)
        plt.text((start_sec + end_sec) / 2, 0.98, phase, ha="center", va="top",
                 fontsize=8, rotation=45, transform=plt.gca().get_xaxis_transform())

    if "artifact_flag" in df.columns:
        for _, row in df.loc[df["artifact_flag"]].iterrows():
            plt.axvspan(row["start_sec"], row["end_sec"], color="red", alpha=0.04)

    for col in cols_to_plot:
        if col in df.columns:
            plt.plot(df["center_sec"], df[col], label=col)

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.show()


# =========================
# MAIN
# =========================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    phase_df = load_phase_table(CSV_FILE)

    data = prepare_data(NPY_FILE, channel_axis=CHANNEL_AXIS, eeg_channels=EEG_CHANNELS)
    if len(CHANNEL_NAMES) != data.shape[1]:
        raise ValueError(
            f"CHANNEL_NAMES length ({len(CHANNEL_NAMES)}) does not match number of EEG channels ({data.shape[1]})."
        )

    data = notch_filter(data, SFREQ, freq=50.0, quality=30.0)
    data = bandpass_filter(data, SFREQ, l_freq=1.0, h_freq=40.0, order=4)
    data = apply_reference(data)

    window_df = compute_window_feature_table(data)
    window_df = assign_phase_to_windows(window_df, phase_df)
    window_df, feature_cols = create_feature_variants(window_df)

    window_df, axis_cols = compute_axes(window_df)

    output_features_csv = RESULTS_DIR / f"{RUN_NAME}_core_features.csv"
    output_pairwise_csv = RESULTS_DIR / f"{RUN_NAME}_feature_pairwise_relevance.csv"
    output_summary_csv = RESULTS_DIR / f"{RUN_NAME}_feature_relevance_summary.csv"
    output_phase_change_csv = RESULTS_DIR / f"{RUN_NAME}_phase_internal_change.csv"
    output_artifact_counts_csv = RESULTS_DIR / f"{RUN_NAME}_artifact_counts.csv"
    output_axes_plot = RESULTS_DIR / f"{RUN_NAME}_axes_over_time.png"
    output_features_plot = RESULTS_DIR / f"{RUN_NAME}_core_features_over_time.png"

    window_df.to_csv(output_features_csv, index=False)

    relevance_features = [c for c in feature_cols if c in window_df.columns and not c.endswith("_z")]
    if ADD_GLOBAL_ZSCORE_COLUMNS:
        relevance_features += [f"{c}_z" for c in relevance_features if f"{c}_z" in window_df.columns]

    pairwise_df = compute_feature_relevance(window_df, relevance_features)
    relevance_summary_df = summarize_relevance(pairwise_df)
    phase_change_df = compute_phase_internal_change(window_df, relevance_features)
    artifact_counts_df = compute_artifact_counts(window_df)

    pairwise_df.to_csv(output_pairwise_csv, index=False)
    relevance_summary_df.to_csv(output_summary_csv, index=False)
    phase_change_df.to_csv(output_phase_change_csv, index=False)
    artifact_counts_df.to_csv(output_artifact_counts_csv, index=False)

    if PLOT_AXES and axis_cols:
        plot_time_series(
            window_df,
            phase_df,
            axis_cols,
            title="Formula-based control axes",
            output_file=output_axes_plot,
            ylabel="axis value",
        )

    if PLOT_BASE_FEATURES:
        plot_time_series(
            window_df,
            phase_df,
            PLOT_FEATURES,
            title="Core EEG features over time",
            output_file=output_features_plot,
            ylabel="feature value",
        )

    print("\nSaved:")
    print(output_features_csv)
    print(output_pairwise_csv)
    print(output_summary_csv)
    print(output_phase_change_csv)
    print(output_artifact_counts_csv)
    if PLOT_AXES and axis_cols:
        print(output_axes_plot)
    if PLOT_BASE_FEATURES:
        print(output_features_plot)

    print("\nTop 20 relevance features:")
    if not relevance_summary_df.empty:
        print(relevance_summary_df.head(20).to_string(index=False))
    else:
        print("No relevance summary computed.")


if __name__ == "__main__":
    main()
