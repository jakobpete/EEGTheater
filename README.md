# EEG Theater Pipeline (TouchDesigner + Python)

This repository contains a compact EEG feature-extraction pipeline developed for a theater / installation context.  
The goal is to derive **stable, stage-safe control parameters** (e.g., alpha/theta markers and motor imagery proxies) from EEG streams and feed them into an interactive system (TouchDesigner).

The codebase supports:

- **Offline validation** on recorded datasets (with annotations/events if present)
- **Online / pseudo-realtime processing** (designed to be embedded in TouchDesigner Python)

---

## Repository Structure

pipeline/
offline_.py
online_.py
README.md
ExampleData/
... EDF files, sample recordings, etc.
eeg_features/
init.py
normalization.py



### `pipeline/ExampleData/`
Sample EEG recordings for offline testing and debugging.  
Typical usage: test your feature logic against known paradigms (e.g., motor imagery datasets).

### `pipeline/eeg_features/`
Small, reusable utilities that are used by both scripts:

- **Normalization / baselining** helpers (z-score based, stage-safe clamping, optional smoothing)
- **Preprocessing utilities** (e.g., average reference, bandpass, light artifact hygiene)  
  *(Exact contents depend on what is currently committed — the intention is to keep this folder as the “portable core.”)*

### Scripts
- **Offline script**  
  Runs sliding-window feature extraction on recorded data and plots features against time (optionally overlaying annotations).
- **Online script**  
  Runs the same feature logic in a pseudo-realtime mode (intended for LSL input and TouchDesigner integration).

---

## What the Pipeline Computes

### 1) Relaxation / Arousal / Internal Orientation
Typical robust markers (baseline-relative):

- **Theta power (4–8 Hz)**
- **Alpha power (8–12 Hz)**
- **Theta/Alpha ratio**
- Optional smoothing and clamping for stable control signals

Interpretation is **state-level**, not diagnostic:
- Alpha/theta dynamics can indicate shifts toward calm vs active-internal modes.

### 2) Motor Imagery (MI)
Motor imagery is modeled via **ERD-like behavior** (baseline-relative decrease) in:

- **Mu band (8–13 Hz) over C3/C4**
- **Beta band (13–30 Hz) over C3/C4** (optional second evidence channel)

Derived indices:
- `mi_global`: global MI engagement proxy (negative relative shifts = ERD-like)
- `mi_lateral`: lateralization proxy (C3 vs C4 difference)

---

## Preprocessing Philosophy (Stage-Safe)

This project deliberately aims for **robustness and controllability** over maximal offline-clean neuroscience.

Typical minimal preprocessing steps:
1. **(Optional) per-channel de-mean**
2. **Average reference** (improves spatial interpretability, especially MI)
3. **Bandpass 1–40 Hz** (stabilizes bandpower features)
4. **Baseline normalization** (critical for comparability and drift resistance)
5. **Smoothing and clamping** (prevents unstable jumps in output parameters)

If you have access to a **vendor “pre-screened / ASR” stream**, it is generally preferred for online use.

---

## Setup

### Requirements
- Python 3.10+ recommended (TouchDesigner uses its own Python runtime)
- Core Python packages typically used:
  - `numpy`
  - `scipy` (if bandpass/notch is used outside MNE)
  - `matplotlib` (offline plots)
  - `pylsl` (online/LSL input if applicable)

Install (example):
```bash
pip install numpy scipy matplotlib pylsl
