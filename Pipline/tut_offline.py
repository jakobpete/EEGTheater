"""
offline_feature_test_with_baseline_file.py

Ziel (OFFLINE):
- Baseline aus separater EDF-Datei gewinnen (optional, für saubere Offline-Validierung)
- Danach Analyse-EDF durch den gleichen Feature-Loop laufen lassen
- Features:
  - Relax/Arousal: Theta, Alpha, Theta/Alpha (z-score; Ratio zusätzlich geglättet)
  - Motor Imagery: Mu/Beta an C3/C4, mi_global, mi_lateral
- Optional: Plot mit raw.annotations Overlay (T0/T1/T2), um Pipeline zu validieren

WICHTIG FÜR SPÄTER ONLINE/LIVE:
- Alles, was "Baseline-File vorfüttern" ist, ist OFFLINE-spezifisch.
- Live hast du statt Baseline-File typischerweise eine Baseline-Zeit am Anfang (z.B. 20s).
- Für Live ersetzt du das Laden von EDF durch LSL-Ringbuffer (kommt später).

In diesem Skript sind OFFLINE-spezifische Blöcke mit:
    ### OFFLINE-ONLY ###
markiert und es steht jeweils, was du für LIVE ersetzen/entfernen musst.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

from eeg_features.normalization import BaselineNormalizer


# ======================================================
# 0) DATEIPFADE (OFFLINE-ONLY)
# ======================================================
# ### OFFLINE-ONLY ###
# Für Offline-Tests: Baseline aus einem (möglichst ruhigen) Run laden
# und danach ein anderes Run-File analysieren.

BASELINE_EDF_PATH = "/Users/jakobnieder/Documents/BurgtheaterEEG/pythonProject/Pipline/ExampleData/S001R01.edf"
ANALYSIS_EDF_PATH = "/Users/jakobnieder/Documents/BurgtheaterEEG/pythonProject/Pipline/ExampleData/S001R04.edf"

# Für LIVE:
# - Diese Pfade entfallen komplett.
# - Stattdessen: LSL inlet + Ringbuffer, und Baseline wird aus den ersten Sekunden gesammelt.


# ======================================================
# 1) HELPER: EDF laden (OFFLINE-ONLY)
# ======================================================
def load_raw_edf(path: str) -> mne.io.BaseRaw:
    """
    Lädt EDF als MNE Raw, wählt EEG-Kanäle, säubert Kanalnamen.

    ### OFFLINE-ONLY ###
    In LIVE ersetzt du diese Funktion durch LSL-Inlet/Ringbuffer.
    """
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.pick_types(meg=False, eeg=True, eog=False, stim=False)
    raw.rename_channels(lambda ch: ch.replace(".", "").upper())
    return raw


# ======================================================
# 2) Bandpower (gleich für Offline und später Live)
# ======================================================
def compute_bandpower(data_window: np.ndarray, fmin: float, fmax: float, sfreq: float) -> np.ndarray:
    """
    Berechnet mittlere Bandpower pro Kanal im Frequenzband [fmin, fmax].

    data_window: shape (n_channels, n_times)

    Hinweis:
    - n_fft und n_per_seg werden explizit auf Fensterlänge gesetzt
      -> reproduzierbar und transparent
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
    return np.log10(psd.mean(axis=1) + 1e-15)


# ======================================================
# 3) Fensterparameter (Offline & Live identisch gedacht)
# ======================================================
WINDOW_SEC = 2.0
STEP_SEC = 0.5

# Baseline-Dauer ist hier "virtuell" über warmup_windows definiert.
# Wenn du Baseline aus separatem File fütterst, muss dieses File lang genug sein.
BASELINE_SEC = 20.0
warmup_windows = max(5, int(BASELINE_SEC / STEP_SEC))

print(f"WINDOW={WINDOW_SEC}s | STEP={STEP_SEC}s | BASELINE_SEC≈{BASELINE_SEC}s => warmup_windows={warmup_windows}")


# ======================================================
# 4) Normalizer initialisieren (Offline & Live identisch)
# ======================================================
# Relax/Arousal
bn_theta = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
bn_alpha = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
bn_ratio = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=0.2)  # Ratio geglättet


# ======================================================
# 5) MI Kanal-Setup hängt an Kanalnamen (kommt aus Raw/LSL)
# ======================================================
def find_ch(ch_names: list[str], name: str) -> int | None:
    """
    Gibt Index des Kanals zurück oder None.
    (Type-Hint: int | None = 'entweder int oder None')
    """
    name = name.upper()
    return ch_names.index(name) if name in ch_names else None


# ======================================================
# 6) Baseline-Fütterung aus separatem Baseline-File (OFFLINE-ONLY)
# ======================================================
def feed_baseline_from_data(
    data: np.ndarray,
    sfreq: float,
    window_samples: int,
    step_samples: int,
    ch_names: list[str],
    # Relax/Arousal Normalizer
    bn_theta: BaselineNormalizer,
    bn_alpha: BaselineNormalizer,
    bn_ratio: BaselineNormalizer,
    # optional MI Normalizer
    bn_mu_C3: BaselineNormalizer | None = None,
    bn_mu_C4: BaselineNormalizer | None = None,
    bn_beta_C3: BaselineNormalizer | None = None,
    bn_beta_C4: BaselineNormalizer | None = None,
) -> None:
    """
    Füttert die Normalizer, bis sie "ready" sind.

    ### OFFLINE-ONLY ###
    Für LIVE:
    - Du entfernst diesen kompletten Schritt.
    - Stattdessen läuft dein Live-Loop sofort, und du rufst update()
      auf den Normalizern während der ersten BASELINE_SEC Sekunden auf.
    """
    idx_C3 = find_ch(ch_names, "C3")
    idx_C4 = find_ch(ch_names, "C4")
    has_mi = (idx_C3 is not None) and (idx_C4 is not None) and (bn_mu_C3 is not None)

    start = 0
    n_samples = data.shape[1]

    while start + window_samples <= n_samples:
        stop = start + window_samples
        window = data[:, start:stop]

        # Relax/Arousal baseline füllen
        theta_mean = compute_bandpower(window, 4, 8, sfreq).mean()
        alpha_mean = compute_bandpower(window, 8, 12, sfreq).mean()
        ratio = theta_mean / (alpha_mean + 1e-15)

        bn_theta.update(theta_mean)
        bn_alpha.update(alpha_mean)
        bn_ratio.update(ratio)

        # MI baseline füllen (falls vorhanden)
        if has_mi:
            mu = compute_bandpower(window, 8, 13, sfreq)
            beta = compute_bandpower(window, 13, 30, sfreq)

            bn_mu_C3.update(float(mu[idx_C3]))
            bn_mu_C4.update(float(mu[idx_C4]))
            bn_beta_C3.update(float(beta[idx_C3]))
            bn_beta_C4.update(float(beta[idx_C4]))

        # Abbruchbedingung: sobald Ratio ready ist (Relax/Arousal ready)
        # und falls MI aktiv: auch Mu/Beta ready
        relax_ready = bn_ratio.ready
        mi_ready = True
        if has_mi:
            mi_ready = bn_mu_C3.ready and bn_mu_C4.ready and bn_beta_C3.ready and bn_beta_C4.ready

        if relax_ready and mi_ready:
            break

        start += step_samples


# ======================================================
# 7) OFFLINE-ONLY: Baseline-File laden und Normalizer vorfüttern
# ======================================================
# ### OFFLINE-ONLY ###
baseline_raw = load_raw_edf(BASELINE_EDF_PATH)
baseline_raw.set_eeg_reference("average", projection=False)
baseline_raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)

sfreq = float(baseline_raw.info["sfreq"])
ch_names = list(baseline_raw.ch_names)

window_samples = int(WINDOW_SEC * sfreq)
step_samples = int(STEP_SEC * sfreq)

print("\n=== BASELINE FILE ===")
print("Baseline file:", BASELINE_EDF_PATH)
print("sfreq:", sfreq)
print("channels:", ch_names)
print("=====================\n")

baseline_data, _ = baseline_raw.get_data(return_times=True)

# MI Normalizer pro Kanal/Band (nur wenn C3/C4 existieren)
idx_C3 = find_ch(ch_names, "C3")
idx_C4 = find_ch(ch_names, "C4")
HAS_MI = (idx_C3 is not None) and (idx_C4 is not None)
print(f"MI Kanäle im Baseline-File: C3={idx_C3}, C4={idx_C4} | HAS_MI={HAS_MI}")

bn_mu_C3 = bn_mu_C4 = bn_beta_C3 = bn_beta_C4 = None
if HAS_MI:
    bn_mu_C3   = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
    bn_mu_C4   = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
    bn_beta_C3 = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)
    bn_beta_C4 = BaselineNormalizer(warmup_windows=warmup_windows, clip=3.0, smooth_alpha=None)

print("Füttere Baseline-Normalizer aus Baseline-File ...")
feed_baseline_from_data(
    data=baseline_data,
    sfreq=sfreq,
    window_samples=window_samples,
    step_samples=step_samples,
    ch_names=ch_names,
    bn_theta=bn_theta,
    bn_alpha=bn_alpha,
    bn_ratio=bn_ratio,
    bn_mu_C3=bn_mu_C3,
    bn_mu_C4=bn_mu_C4,
    bn_beta_C3=bn_beta_C3,
    bn_beta_C4=bn_beta_C4,
)

if not bn_ratio.ready:
    raise RuntimeError("Baseline konnte nicht gefüllt werden: Baseline-File zu kurz oder Parameter unpassend.")

if HAS_MI and not (bn_mu_C3.ready and bn_mu_C4.ready):
    raise RuntimeError("MI Baseline konnte nicht gefüllt werden: Baseline-File zu kurz oder MI-Kanäle fehlen.")

print("Baseline ready (aus Baseline-File). Starte Analyse.\n")

# Für LIVE:
# - Du entfernst alles ab "baseline_raw = ..." bis hier.
# - Und gehst direkt in den Live-Loop, wo update() die Baseline während der ersten Sekunden sammelt.


# ======================================================
# 8) OFFLINE-ONLY: Analyse-File laden
# ======================================================
# ### OFFLINE-ONLY ###
raw = load_raw_edf(ANALYSIS_EDF_PATH)

raw.set_eeg_reference("average", projection=False)
raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)

print("Analyse file:", ANALYSIS_EDF_PATH)

# Sicherheitscheck: gleiche Kanalreihenfolge wie Baseline-File
if list(raw.ch_names) != ch_names:
    raise RuntimeError(
        "Channel mismatch zwischen Baseline-File und Analyse-File.\n"
        f"Baseline channels: {ch_names}\n"
        f"Analysis channels:  {list(raw.ch_names)}\n"
        "Lösung: gleiche Runs/Setup nutzen oder Kanäle explizit reordnen."
    )

data, times = raw.get_data(return_times=True)
n_channels, n_samples = data.shape


# ======================================================
# 9) OFFLINE: Annotations (T0/T1/T2) extrahieren für Plot-Overlay
# ======================================================
# ### OFFLINE-ONLY (für Validierung sehr hilfreich) ###
annotations = raw.annotations
print("Annotations:", annotations)

event_blocks = []
for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
    event_blocks.append({"onset": float(onset), "duration": float(duration), "label": str(desc)})


def overlay_annotations(ax, event_blocks, color_map=None, alpha=0.12):
    """
    Zeichnet Annotationen als farbige Zeitblöcke, ohne Legend-Explosion:
    - jedes Label (T0/T1/T2) wird nur einmal gelabelt
    """
    if color_map is None:
        color_map = {"T0": "gray", "T1": "blue", "T2": "red"}

    seen = set()
    for ev in event_blocks:
        onset = ev["onset"]
        dur = ev["duration"]
        label = ev["label"]
        color = color_map.get(label, "black")

        lbl = label if label not in seen else None
        seen.add(label)

        ax.axvspan(onset, onset + dur, alpha=alpha, color=color, label=lbl)


# ======================================================
# 10) Analyse-Loop (dieser Teil ist später "live-portierbar")
# ======================================================
# Für LIVE:
# - data/times kommen nicht aus EDF, sondern aus Ringbuffer-Fenstern.
# - Der innere Teil (compute_bandpower + update + Indizes) bleibt sehr ähnlich.

time_list = []

# Relax/Arousal outputs
theta_z_list = []
alpha_z_list = []
ratio_smooth_list = []

# MI outputs
mi_global_list = []
mi_lateral_list = []
mi_global_smooth = None
mi_lateral_smooth = None

def exp_smooth(prev, new, alpha=0.2):
    return new if prev is None else (1 - alpha) * prev + alpha * new


# Optional: raw channel z-lists (zum Debuggen)
mu_C3_z_list = []
mu_C4_z_list = []

# Thresholds für MI-States (optional)
MI_ACTIVE_THR = -1.0
MI_LATERAL_THR = 0.5

start = 0
window_idx = 0

while start + window_samples <= n_samples:
    stop = start + window_samples
    window_data = data[:, start:stop]
    t_center = float(times[start:stop].mean())

    # -------- Relax/Arousal --------
    theta_mean = float(compute_bandpower(window_data, 4, 8, sfreq).mean())
    alpha_mean = float(compute_bandpower(window_data, 8, 12, sfreq).mean())
    ratio = theta_mean / (alpha_mean + 1e-15)

    theta_z = bn_theta.update(theta_mean)
    alpha_z = bn_alpha.update(alpha_mean)
    ratio_smooth = bn_ratio.update(ratio)

    # Da wir Baseline schon vorgefüttert haben, sollte hier nichts mehr None sein.
    if ratio_smooth is None:
        raise RuntimeError("Unerwartet: bn_ratio ist nicht ready, obwohl Baseline vorgefüttert wurde.")

    # -------- Motor Imagery --------
    mi_global = np.nan
    mi_lateral = np.nan

    if HAS_MI:
        mu = compute_bandpower(window_data, 8, 13, sfreq)
        mu_C3 = float(mu[idx_C3])
        mu_C4 = float(mu[idx_C4])

        mu_C3_z = bn_mu_C3.update(mu_C3)
        mu_C4_z = bn_mu_C4.update(mu_C4)

        if (mu_C3_z is None) or (mu_C4_z is None):
            raise RuntimeError("Unerwartet: MI Normalizer nicht ready, obwohl Baseline vorgefüttert wurde.")


        beta = compute_bandpower(window_data, 13, 30, sfreq)
        beta_C3 = float(beta[idx_C3])
        beta_C4 = float(beta[idx_C4])

        beta_C3_z = bn_beta_C3.update(beta_C3)
        beta_C4_z = bn_beta_C4.update(beta_C4)

        if (beta_C3_z is None) or (beta_C4_z is None):
            raise RuntimeError("Unerwartet: Beta-Normalizer nicht ready, obwohl Baseline vorgefüttert wurde.")

        # MI Indizes
        mi_global = float(np.mean([mu_C3_z, mu_C4_z, beta_C3_z, beta_C4_z]))
        mi_lateral = float((mu_C3_z + beta_C3_z) - (mu_C4_z + beta_C4_z))



        mi_global_smooth = exp_smooth(mi_global_smooth, mi_global, alpha=0.2)
        mi_lateral_smooth = exp_smooth(mi_lateral_smooth, mi_lateral, alpha=0.2)


        mu_C3_z_list.append(mu_C3_z)
        mu_C4_z_list.append(mu_C4_z)

    # -------- speichern --------
    time_list.append(t_center)
    theta_z_list.append(theta_z)
    alpha_z_list.append(alpha_z)
    ratio_smooth_list.append(ratio_smooth)

    # mi_global_list.append(mi_global)
    # mi_lateral_list.append(mi_lateral)
    mi_global_list.append(mi_global_smooth)
    mi_lateral_list.append(mi_lateral_smooth)


    # Debug Print (sparsam halten, offline ok)
    if window_idx % 5 == 0:
        print(
            f"t={t_center:6.2f}s | ratio_smooth={ratio_smooth:6.2f} | "
            f"mi_global={mi_global:6.2f} mi_lateral={mi_lateral:6.2f}"
        )

    start += step_samples
    window_idx += 1


# ======================================================
# 11) Plot: MI global + MI lateral mit Event-Overlay
# ======================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# MI global
ax1.plot(time_list, mi_global_list, label="MI global (Mu z mean)")
ax1.axhline(0, linestyle="--", linewidth=1, label="0 = Baseline")
ax1.axhline(MI_ACTIVE_THR, linestyle=":", linewidth=1, label="MI threshold")
overlay_annotations(ax1, event_blocks, color_map={"T0": "gray", "T1": "blue", "T2": "red"})
ax1.set_ylabel("MI global (z)")
ax1.legend(loc="upper right")

# MI lateral
ax2.plot(time_list, mi_lateral_list, linestyle="--", label="MI lateral (Mu C3_z - C4_z)")
ax2.axhline(0, linestyle="--", linewidth=1, label="0 = keine Lateralisierung")
overlay_annotations(ax2, event_blocks, color_map={"T0": "gray", "T1": "blue", "T2": "red"})
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("MI lateral (z-diff)")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()



def mean_in_label(label):
    vals = []
    for ev in event_blocks:
        if ev["label"] != label:
            continue
        a, b = ev["onset"], ev["onset"] + ev["duration"]
        vals.extend([v for t, v in zip(time_list, mi_global_list) if a <= t <= b and np.isfinite(v)])
    return np.mean(vals), np.std(vals), len(vals)

for lab in ["T0", "T1", "T2"]:
    m, s, n = mean_in_label(lab)
    print(lab, m, s, n)

# ======================================================
# 12) Was du für LIVE zurückbauen / ersetzen musst (Kurzliste)
# ======================================================
"""
### WENN DU AUF ONLINE/LIVE UMSTELLST ###

A) Entfernen/ersetzen (OFFLINE-ONLY):
- BASELINE_EDF_PATH / ANALYSIS_EDF_PATH
- load_raw_edf()
- baseline_raw laden + baseline_data extrahieren
- feed_baseline_from_data(...) (komplett)
- raw = load_raw_edf(ANALYSIS_EDF_PATH) und data/times aus EDF
- Annotation-Overlay (optional; live hast du i.d.R. keine T0/T1/T2 Labels)

B) Was bleibt (fast) identisch:
- compute_bandpower()
- BaselineNormalizer-Objekte (bn_theta, bn_alpha, bn_ratio, bn_mu_C3, bn_mu_C4, ...)
- Feature-Logik im Loop:
    - theta/alpha/ratio berechnen
    - bn.update(...) anwenden
    - mi_global/mi_lateral ableiten

C) Was du für LIVE statt EDF machst:
- LSL inlet + Ringbuffer
- window_data = get_latest_window(window_samples)
- t_center aus "time.time()" oder aus LSL timestamps
- Baseline wird live gesammelt, indem du in den ersten BASELINE_SEC Sekunden
  bn.update(...) aufrufst und solange Outputs ggf. als "calibrating" behandelst.
"""
print("idx_C3:", idx_C3, "idx_C4:", idx_C4, "HAS_MI:", HAS_MI)
print("ch_names:", ch_names)
