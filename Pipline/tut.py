"""
offline_feature_test_refactored.py

Offline-Prototyp:
- EDF laden
- Sliding-Window-Bandpower
- Initiale Baseline + Z-Score-Normalisierung
- Optionales Smoothing für Ratio

Kein LSL, kein TouchDesigner – reine Feature-Logik.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

from eeg_features.normalization import BaselineNormalizer


# ======================================================
# 1) EEG laden (offline)
# ======================================================

raw = mne.io.read_raw_edf(
    "/Users/jakobnieder/Documents/BurgtheaterEEG/pythonProject/Pipline/ExampleData/S001R04.edf",
    preload=True,
)

# =============================
# Validierung 
# =============================
events, event_id = mne.events_from_annotations(raw)
annotations = raw.annotations

event_blocks = []

for onset, duration, desc in zip(
    annotations.onset,
    annotations.duration,
    annotations.description
):
    event_blocks.append({
        "onset": onset,            # Sekunden
        "duration": duration,      # Sekunden
        "label": desc              # "T0", "T1", "T2"
    })

print(event_blocks[:5])

def overlay_annotations(ax, event_blocks, color_map=None, alpha=0.15):
    if color_map is None:
        color_map = {"T0": "gray", "T1": "blue", "T2": "red"}

    # Merkt sich, ob ein Label schon einmal in die Legend aufgenommen wurde
    seen = set()

    for ev in event_blocks:
        onset = ev["onset"]
        dur = ev["duration"]
        label = ev["label"]
        color = color_map.get(label, "black")

        # Nur beim ersten Auftreten ein label setzen, danach label=None
        lbl = label if label not in seen else None
        seen.add(label)

        ax.axvspan(onset, onset + dur, alpha=alpha, color=color, label=lbl)



# ======================================================
# Validierung nach oben
# ======================================================


raw.pick_types(meg=False, eeg=True, eog=False, stim=False)
raw.rename_channels(lambda ch: ch.replace(".", "").upper())

sfreq = raw.info["sfreq"]
ch_names = raw.ch_names

print(f"Samplingrate: {sfreq} Hz")
print(f"Kanäle: {ch_names}")

data, times = raw.get_data(return_times=True)
n_channels, n_samples = data.shape

# ======================================================
# 2) load Annotations for Data
# ======================================================
#annotations = mne.io.read_raw_

# ======================================================
# 2) Fensterparameter
# ======================================================

WINDOW_SEC = 2.0
STEP_SEC = 0.5

window_samples = int(WINDOW_SEC * sfreq)
step_samples = int(STEP_SEC * sfreq)

print(f"WINDOW={WINDOW_SEC}s ({window_samples} samples)")
print(f"STEP={STEP_SEC}s ({step_samples} samples)")


# ======================================================
# 3) Baseline-Normalizer initialisieren
# ======================================================

BASELINE_SEC = 20.0
# ungefähr 20 Sekunden Fenster-Updates
warmup_windows = max(5, int(BASELINE_SEC / STEP_SEC))
default_clip= 5

# consciousness
bn_theta = BaselineNormalizer(warmup_windows=warmup_windows, clip=default_clip, smooth_alpha=None,)
bn_alpha = BaselineNormalizer(warmup_windows=warmup_windows, clip=default_clip, smooth_alpha=None,)
bn_ratio = BaselineNormalizer(warmup_windows=warmup_windows, clip=default_clip,smooth_alpha=0.2, )  # Ratio wird geglättet

print(f"Baseline: {BASELINE_SEC}s ≈ {warmup_windows} Fenster")


# ======================================================
# 3b) Motor Imagery Kanal-Setup + Normalizer
# ======================================================

def find_ch(name: str) -> int | None:
    name = name.upper()
    return ch_names.index(name) if name in ch_names else None

idx_C3 = find_ch("C3")
idx_C4 = find_ch("C4")

HAS_MI = (idx_C3 is not None) and (idx_C4 is not None)
print(f"MI Kanäle: C3={idx_C3}, C4={idx_C4} | HAS_MI={HAS_MI}")

# Normalizer pro Kanal & Band (nur wenn Kanäle existieren)
if HAS_MI:
    bn_mu_C3   = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
    bn_mu_C4   = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
    bn_beta_C3 = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
    bn_beta_C4 = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)


# ======================================================
# 4) Bandpower-Hilfsfunktion
# ======================================================

def compute_bandpower(data_window, fmin, fmax, sfreq):
    """
    Berechnet mittlere Bandpower pro Kanal für ein Fenster.
    """
    psd, freqs = mne.time_frequency.psd_array_welch(
        data_window,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=data_window.shape[1],
        n_per_seg=data_window.shape[1],
        average="mean",
        verbose=False,
    )
    return psd.mean(axis=1)


# ======================================================
# 5) Sliding-Window Loop
# ======================================================

time_list = []

# Relax/Arousal Listen (z)
theta_z_list = []
alpha_z_list = []
ratio_smooth_list = []

# MI Listen (z/ERD + states)
mu_C3_z_list = []
mu_C4_z_list = []
beta_C3_z_list = []
beta_C4_z_list = []
mi_global_list = []
mi_lateral_list = []
mi_active_list = []
mi_left_list = []
mi_right_list = []

# Thresholds (tunable)
MI_ACTIVE_THR = -1.0   # global ERD-Schwelle (negativ = ERD)
MI_LATERAL_THR = 0.5   # Lateralisierungs-Schwelle (z-Differenz)

start = 0
window_idx = 0


while start + window_samples <= n_samples:
    stop = start + window_samples
    window_data = data[:, start:stop]
    t_center = times[start:stop].mean()

    # --------------------------------------------------
    # A) Relax/Arousal Features
    # --------------------------------------------------
    theta = compute_bandpower(window_data, 4, 8, sfreq)
    alpha = compute_bandpower(window_data, 8, 12, sfreq)

    theta_mean = theta.mean()
    alpha_mean = alpha.mean()
    theta_alpha = theta_mean / (alpha_mean + 1e-15)

    theta_z = bn_theta.update(theta_mean)
    alpha_z = bn_alpha.update(alpha_mean)
    ratio_smooth = bn_ratio.update(theta_alpha)

    # --------------------------------------------------
    # B) Motor Imagery Features (Mu/Beta an C3/C4)
    #     - getrennte Kanäle
    #     - ERD-Lesbarkeit: negative z-Werte = unter Baseline (typisch MI)
    #     - Zustände: MI_ACTIVE / MI_LEFT / MI_RIGHT
    # --------------------------------------------------
    mu_C3_z = mu_C4_z = beta_C3_z = beta_C4_z = None
    mi_global = mi_lateral = None
    MI_ACTIVE = MI_LEFT = MI_RIGHT = None

    if HAS_MI:
        # Mu (8–13) und Beta (13–30) pro Kanal
        mu   = compute_bandpower(window_data, 8, 13, sfreq)
        beta = compute_bandpower(window_data, 13, 30, sfreq)

        mu_C3 = float(mu[idx_C3])
        mu_C4 = float(mu[idx_C4])
        beta_C3 = float(beta[idx_C3])
        beta_C4 = float(beta[idx_C4])

        # Baseline + Z-Score pro Kanal
        mu_C3_z   = bn_mu_C3.update(mu_C3)
        mu_C4_z   = bn_mu_C4.update(mu_C4)
        beta_C3_z = bn_beta_C3.update(beta_C3)
        beta_C4_z = bn_beta_C4.update(beta_C4)

        # Sobald Baseline fertig: MI-Indizes berechnen
        # (Wir orientieren uns hier primär an Mu; Beta ist optional parallel auswertbar.)
        if (mu_C3_z is not None) and (mu_C4_z is not None):
            # Globaler MI-Index (negativ = ERD = MI)
            mi_global = float(np.mean([mu_C3_z, mu_C4_z]))

            # Lateralisierung (Differenz)
            # Vorzeichen-Interpretation hängt davon ab, welche Hand imaginiert wird;
            # als technische Größe ist das stabil und für Mapping ausreichend.
            mi_lateral = float(mu_C3_z - mu_C4_z)

            # Diskrete Zustände
            MI_ACTIVE = (mi_global < MI_ACTIVE_THR)
            MI_LEFT   = MI_ACTIVE and (mi_lateral > +MI_LATERAL_THR)
            MI_RIGHT  = MI_ACTIVE and (mi_lateral < -MI_LATERAL_THR)

    # --------------------------------------------------
    # C) Baseline-Warmup Handling
    # --------------------------------------------------
    # Wir wollen nicht „halb-bereite“ Werte plotten.
    # Für Relax/Arousal warten wir auf ratio_smooth (weil der Normalizer dort smoothing macht).
    # Für MI warten wir auf mu_C3_z/mu_C4_z (wenn MI überhaupt aktiv ist).
    relax_ready = (ratio_smooth is not None)
    mi_ready = (not HAS_MI) or ((mu_C3_z is not None) and (mu_C4_z is not None))

    if not relax_ready or not mi_ready:
        if window_idx % 5 == 0:
            msg = f"[BASELINE] sammle... ratio={len(bn_ratio.values)}/{bn_ratio.warmup_windows}"
            if HAS_MI:
                msg += f" | muC3={len(bn_mu_C3.values)}/{bn_mu_C3.warmup_windows}"
            print(msg)

        start += step_samples
        window_idx += 1
        continue

    # --------------------------------------------------
    # D) Speichern + Output (ab Baseline-ready)
    # --------------------------------------------------
    time_list.append(t_center)
    theta_z_list.append(theta_z)
    alpha_z_list.append(alpha_z)
    ratio_smooth_list.append(ratio_smooth)

    if HAS_MI:
        mu_C3_z_list.append(mu_C3_z)
        mu_C4_z_list.append(mu_C4_z)
        beta_C3_z_list.append(beta_C3_z)
        beta_C4_z_list.append(beta_C4_z)
        mi_global_list.append(mi_global)
        mi_lateral_list.append(mi_lateral)
        mi_active_list.append(int(MI_ACTIVE))
        mi_left_list.append(int(MI_LEFT))
        mi_right_list.append(int(MI_RIGHT))

    # Lesbare Debug-Ausgabe
    out = (
        f"t={t_center:6.2f}s | "
        f"Theta_z={theta_z:6.2f} | Alpha_z={alpha_z:6.2f} | Ratio_smooth={ratio_smooth:6.2f}"
    )
    if HAS_MI:
        out += (
            f" || MI: muC3_z={mu_C3_z:6.2f} muC4_z={mu_C4_z:6.2f} "
            f"mi_global={mi_global:6.2f} mi_lat={mi_lateral:6.2f} "
            f"ACTIVE={MI_ACTIVE} LEFT={MI_LEFT} RIGHT={MI_RIGHT}"
        )
    print(out)

    start += step_samples
    window_idx += 1


# ======================================================
# 6) Plot
# ======================================================

# plt.figure(figsize=(12, 8))

# plt.subplot(4, 1, 1)
# plt.plot(time_list, theta_z_list, label="Theta (z)")
# plt.plot(time_list, alpha_z_list, label="Alpha (z)")
# plt.legend()
# plt.title("Relax/Arousal: Theta & Alpha (z)")

# plt.subplot(4, 1, 2)
# plt.plot(time_list, ratio_smooth_list, label="Theta/Alpha (smoothed z)")
# plt.legend()
# plt.title("Relax/Arousal: Theta/Alpha (z, smoothed)")

# plt.subplot(4, 1, 3)
# if HAS_MI:
#     plt.plot(time_list, mu_C3_z_list, label="Mu C3 (z)")
#     plt.plot(time_list, mu_C4_z_list, label="Mu C4 (z)")
#     plt.legend()
#     plt.title("Motor Imagery: Mu ERD (z) – negative Werte = ERD")
# else:
#     plt.text(0.1, 0.5, "MI: C3/C4 nicht vorhanden", transform=plt.gca().transAxes)

# plt.subplot(4, 1, 4)
# if HAS_MI:
#     plt.plot(time_list, mi_global_list, label="MI global (Mu z mean)")
#     plt.plot(time_list, mi_lateral_list, label="MI lateral (Mu C3_z - C4_z)")
#     plt.legend()
#     plt.title("Motor Imagery: Indizes (global/lateral)")
# else:
#     plt.text(0.1, 0.5, "MI: C3/C4 nicht vorhanden", transform=plt.gca().transAxes)

# plt.tight_layout()
# plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Oben: global
ax1.plot(time_list, mi_global_list, label="MI global (Mu z mean)")
ax1.axhline(0, linestyle="--", linewidth=1)
ax1.axhline(-1.0, linestyle=":", linewidth=1, label="MI threshold")
overlay_annotations(ax1, event_blocks, color_map={"T0": "gray", "T1": "blue", "T2": "red"})
ax1.set_ylabel("MI global (z)")
ax1.legend(loc="upper right")

# Unten: lateral
ax2.plot(time_list, mi_lateral_list, label="MI lateral (Mu C3_z - C4_z)", linestyle="--")
ax2.axhline(0, linestyle="--", linewidth=1)
overlay_annotations(ax2, event_blocks, color_map={"T0": "gray", "T1": "blue", "T2": "red"})
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("MI lateral (z-diff)")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

