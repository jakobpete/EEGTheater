import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal


# =========================
# CONFIG
# =========================
NPY_FILE = "passiveBCI/testruns/data/Rec_Phil_28_03_2026_RUN_1_001.npy"
CSV_FILE = "passiveBCI/testruns/tryouts/Tryouts_28_03_2026-Sheet1.csv"
SFREQ = 250.0
CHANNEL_AXIS = 1
EEG_CHANNELS = list(range(1, 17))  # drop empty channel 0

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

APPLY_AVERAGE_REFERENCE = True
APPLY_LOG_BANDPOWER = True
LOG_EPS = 1e-12



# Sweet spot from analysis
WINDOW_SEC = 6.0
STEP_SEC = 2.0
FEATURE_CONTEXT_RADIUS = 2

# optional artifact handling
ARTIFACT_MODE = "suppress"   # "none", "mask", "suppress", "drop"
BLINK_ROI_NAME = "frontal"
BLINK_THRESHOLD_UV = 100.0
GLOBAL_ARTIFACT_THRESHOLD_UV = 200.0

# plot output
OUTPUT_PLOT = Path("passiveBCI/testruns/results/control_axes_over_time.png")
OUTPUT_AXES_CSV = Path("passiveBCI/testruns/results/control_axes_over_time.csv")

# optional phase shading
EXCLUDE_PHASE_KEYWORDS = ["Pause / Prepare"]
OPTIONAL_EXCLUDE_EXACT = set()

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


# =========================
# HELPERS
# =========================
def load_phase_table(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]

    phase_col = ""
    if phase_col not in df.columns:
        raise ValueError(
            f"Could not find the phase name column. Found columns: {df.columns.tolist()}"
        )

    df = df.rename(columns={
        phase_col: "phase",
        "Dauer": "duration_min",
        "T": "end_time_min",
    })

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
                                    mode: str = "suppress") -> pd.DataFrame:
    if "artifact_flag" not in df.columns:
        return df

    if mode == "none" or mode == "mask":
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
                if len(neighbor_vals) > 0:
                    out.loc[i, col] = float(np.median(neighbor_vals))

        return out

    raise ValueError(f"Unknown ARTIFACT_MODE: {mode}")


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
                                 artifact_mode: str = "none") -> pd.DataFrame:
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

    # local context features over neighboring windows
    feature_cols = [
        "theta", "alpha", "beta",
        "theta_frontal", "alpha_parietal", "beta_central",
    ]

    for col in feature_cols:
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=float)
        local_mean = []
        local_var = []

        for i in range(len(values)):
            start = max(0, i - FEATURE_CONTEXT_RADIUS)
            end = min(len(values), i + FEATURE_CONTEXT_RADIUS + 1)
            local_vals = values[start:end]
            local_mean.append(float(np.mean(local_vals)))
            local_var.append(float(np.var(local_vals)))

        df[f"{col}_local_mean"] = local_mean
        df[f"{col}_local_var"] = local_var

    # ratios
    # IMPORTANT:
    # theta/alpha ratios on log-transformed powers are not physically meaningful as divisions.
    # Since powers were log10-transformed above, the meaningful relative measure is a DIFFERENCE,
    # i.e. log(theta) - log(alpha) = log(theta / alpha).
    df["theta_alpha_log_ratio"] = df["theta"] - df["alpha"]
    df["alpha_beta_log_ratio"] = df["alpha"] - df["beta"]
    df["theta_frontal_alpha_parietal_log_ratio"] = df["theta_frontal"] - df["alpha_parietal"]
    df["theta_frontal_alpha_parietal_log_ratio_local"] = (
        df["theta_frontal_local_mean"] - df["alpha_parietal_local_mean"]
    )

    # Keep the old names as aliases for backward compatibility in downstream code/plots
    df["theta_alpha_ratio"] = df["theta_alpha_log_ratio"]
    df["alpha_beta_ratio"] = df["alpha_beta_log_ratio"]
    df["theta_frontal_alpha_parietal_ratio"] = df["theta_frontal_alpha_parietal_log_ratio"]
    df["theta_frontal_alpha_parietal_ratio_local"] = df["theta_frontal_alpha_parietal_log_ratio_local"]

    feature_cols = [
        "theta", "alpha", "beta", "gamma",
        "theta_frontal", "alpha_parietal", "beta_central",
        "theta_alpha_ratio", "alpha_beta_ratio",
        "theta_frontal_alpha_parietal_ratio",
        "theta_frontal_alpha_parietal_ratio_local",
        "theta_local_mean", "alpha_local_mean", "beta_local_mean",
        "theta_local_var", "alpha_local_var", "beta_local_var",
        "theta_frontal_local_mean", "alpha_parietal_local_mean", "beta_central_local_mean",
        "theta_frontal_local_var", "alpha_parietal_local_var", "beta_central_local_var",
    ]

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


def smooth_and_normalize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Smooth each feature first, then z-score normalize globally.
    This creates stable, comparable inputs for axis construction.
    """

    # --- smoothing first ---
    for col in feature_cols:
        if col in df.columns:
            df[col] = smooth_signal(df[col], window=15)

    # --- global z-scoring ---
    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std

    return df


def compute_axes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # -----------------------------
    # DIAGNOSTIC AXES WITH ROBUST THETA-ALPHA DYNAMICS
    # IMPORTANT: assumes input features are already smoothed + z-scored
    # -----------------------------

    # Robust balance: preserve sustained theta > alpha, but reduce sensitivity to end-of-phase theta spikes
    df["theta_alpha_balance_slow"] = df["theta_slow"] - df["alpha"]
    df["theta_alpha_balance_robust"] = (
        0.7 * df["theta_slow"]
        + 0.3 * df["theta"]
        - df["alpha"]
        - 0.25 * df["theta_var_smooth"]
    )

    # Axis 1: Engagement / joint elevation of alpha and theta
    df["beta_axis"] = (
        df["beta"]          
    )

    # Axis 2: Volatility / joint rise with emphasis on theta variance
    df["gamma_axis"] = (
        df["gamma"]
    )

    # Axis 3: Control-like theta-over-alpha balance, but made robust against theta spikes
    df["theta_bp"] = df["theta"]

    axis_cols = ["beta_axis", "gamma_axis", "theta_bp"]
    return df, axis_cols


def zscore_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
    return df


def smooth_signal(x: pd.Series, window: int = 5) -> pd.Series:
    """
    Exponential Moving Average (EMA) smoothing.
    'window' is interpreted as span (approximate smoothing scale).
    This is causal (no future leakage) and suitable for real-time use.
    """
    return x.ewm(span=window, adjust=False).mean()

def detrend_feature(df: pd.DataFrame, col: str, slow_window: int = 31) -> pd.DataFrame:
    """
    Split a feature into a slow trend and a fast component.
    slow_window is in number of windows, not seconds.
    """
    if col not in df.columns:
        return df

    slow_col = f"{col}_slow"
    fast_col = f"{col}_fast"

    df[slow_col] = smooth_signal(df[col], window=slow_window)
    df[fast_col] = df[col] - df[slow_col]
    df[f"{col}_fast_smooth"] = smooth_signal(df[fast_col], window=90)
    # --- GLOBAL theta variance (based on original theta, not fast) ---
    if col == "theta":
        theta_mean = df[col].rolling(window=9, center=True, min_periods=1).mean()
        df["theta_var"] = (df[col] - theta_mean) ** 2

        # smooth for readability
        df["theta_var_smooth"] = smooth_signal(df["theta_var"], window=30)

    return df
def add_smoothed_feature(df: pd.DataFrame, col: str, out_col: str, window: int) -> pd.DataFrame:
    if col in df.columns:
        df[out_col] = smooth_signal(df[col], window=window)
    return df

def plot_axes_and_features(df: pd.DataFrame, phase_df: pd.DataFrame, axis_cols: list[str]):
    features_to_plot = [
        #"theta",
        # "theta_var_smooth",
        "alpha",
        "theta"
        #"theta_alpha_log_ratio",
        # "theta_alpha_balance_slow",
        # "theta_alpha_balance_robust",
        #"alpha_local_var",
        #"alpha_local_var_smooth"

    ]

    phase_names = [p for p in phase_df["phase"].dropna().unique()]
    cmap = plt.get_cmap("tab20")
    phase_to_color = {phase: cmap(i % 20) for i, phase in enumerate(phase_names)}

    # ---------- plot 1: axes ----------
    plt.figure(figsize=(14, 7))

    for _, row in phase_df.iterrows():
        phase = row["phase"]
        start_sec = row["start_time_min"] * 60.0
        end_sec = row["end_time_min"] * 60.0
        plt.axvspan(start_sec, end_sec, color=phase_to_color[phase], alpha=0.08)
        plt.text((start_sec + end_sec) / 2, 2.7, phase, ha="center", va="top", fontsize=8, rotation=45)

    if "artifact_flag" in df.columns:
        for _, row in df.loc[df["artifact_flag"]].iterrows():
            plt.axvspan(row["start_sec"], row["end_sec"], color="red", alpha=0.05)

    for axis_name in axis_cols:
        plt.plot(df["center_sec"], df[axis_name], label=axis_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Z-scored axis value")
    plt.ylim((-2, 5))
    plt.title("Continuous control axes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.show()

    # ---------- plot 2: base features ----------
    plt.figure(figsize=(14, 7))

    for _, row in phase_df.iterrows():
        phase = row["phase"]
        start_sec = row["start_time_min"] * 60.0
        end_sec = row["end_time_min"] * 60.0
        plt.axvspan(start_sec, end_sec, color=phase_to_color[phase], alpha=0.05)

    if "artifact_flag" in df.columns:
        for _, row in df.loc[df["artifact_flag"]].iterrows():
            plt.axvspan(row["start_sec"], row["end_sec"], color="red", alpha=0.05)

    for feat in features_to_plot:
        if feat in df.columns:
            plt.plot(df["center_sec"], df[feat], label=feat)

    plt.xlabel("Time (s)")
    plt.ylabel("Feature value")
    plt.title("Base features over time (robust theta-alpha balance diagnostics)")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = OUTPUT_PLOT.parent / "base_features_over_time.png"
    plt.savefig(out_path, dpi=200)
    plt.show()


def main():
    phase_df = load_phase_table(CSV_FILE)
    analysis_phases = select_analysis_phases(phase_df)

    data = prepare_data(NPY_FILE, channel_axis=CHANNEL_AXIS, eeg_channels=EEG_CHANNELS)
    if len(CHANNEL_NAMES) != data.shape[1]:
        raise ValueError(
            f"CHANNEL_NAMES length ({len(CHANNEL_NAMES)}) does not match number of EEG channels ({data.shape[1]})."
        )

    roi_indices = resolve_roi_indices(CHANNEL_NAMES, ROI_CHANNELS)

    data = notch_filter(data, SFREQ, freq=50.0, quality=30.0)
    data = bandpass_filter(data, SFREQ, l_freq=1.0, h_freq=40.0, order=4)
    if APPLY_AVERAGE_REFERENCE:
        data = average_reference(data)

    window_df = compute_window_feature_table(
        data,
        SFREQ,
        roi_indices=roi_indices,
        artifact_mode=ARTIFACT_MODE,
    )
    window_df = assign_phase_to_windows(window_df, analysis_phases)

    if "artifact_flag" in window_df.columns:
        n_art = int(window_df["artifact_flag"].sum())
        n_total = len(window_df)
        print(f"Artifact windows ({ARTIFACT_MODE}): {n_art}/{n_total} ({100*n_art/n_total:.1f}%)")

    # -----------------------------
    # Step 1: smooth + normalize BASE features
    # -----------------------------
    base_features = [
        "theta", "alpha", "beta",
        "theta_frontal", "alpha_parietal",
        "theta_frontal_alpha_parietal_ratio_local",
        "theta_frontal_alpha_parietal_log_ratio_local",
        "theta_alpha_ratio", "theta_alpha_log_ratio",
        "alpha_local_var", "alpha_local_var_smooth", "theta_frontal_local_var", "beta_local_var",
    ]

    window_df = smooth_and_normalize_features(window_df, base_features)
    window_df = add_smoothed_feature(window_df, "alpha_local_var", "alpha_local_var_smooth", 20)
    # -----------------------------
    # Step 1b: split slow vs fast theta-related trends
    # -----------------------------
    window_df = detrend_feature(window_df, "theta", slow_window=31)
    window_df = detrend_feature(window_df, "theta_frontal_alpha_parietal_ratio_local", slow_window=31)

    # normalize the newly created slow/fast features as well
    extra_features = [
        "theta_slow", "theta_fast", "theta_fast_smooth","theta_var_smooth",
        "theta_frontal_alpha_parietal_ratio_local_slow",
        "theta_frontal_alpha_parietal_ratio_local_fast",
        "theta_frontal_alpha_parietal_ratio_local_fast_smooth",
    ]
    for col in extra_features:
        if col in window_df.columns:
            mean = window_df[col].mean()
            std = window_df[col].std()
            if std > 0:
                window_df[col] = (window_df[col] - mean) / std

    # -----------------------------
    # Step 2: compute axes from normalized features
    # -----------------------------
    window_df, axis_cols = compute_axes(window_df)

    # -----------------------------
    # Step 3: light smoothing on final axes
    # -----------------------------
    #for col in axis_cols:
    #    window_df[col] = smooth_signal(window_df[col], window=5)

    OUTPUT_AXES_CSV.parent.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(OUTPUT_AXES_CSV, index=False)
    plot_axes_and_features(window_df, analysis_phases, axis_cols)

    print(f"Saved axis values to: {OUTPUT_AXES_CSV}")
    print(f"Saved plot to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()