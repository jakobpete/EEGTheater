# EEGTheater

EEGTheater is a modular EEG processing pipeline for real-time **Motor Imagery (MI)** decoding in an artistic / stage performance context (SOLARIS theatre project).

This repository contains **two complementary pipelines**:

- **Pipeline A – PhysioNet-based experiments:** offline benchmarking and rapid prototyping on a public dataset.
- **Pipeline B – Own data (LSL → FIF → training):** cue presentation + recording + training for models intended for stage deployment.

The decoding logic is structured as a **two-stage cascade**:

- **Stage 1:** MI vs Rest (probability `p(MI)`)
- **Stage 2:** MI Type (Fists vs Feet; probability `p(feet)` where class 1 = feet, class 0 = fists)

Stage 2 is intended to be evaluated **only when Stage 1 indicates MI** (gate).

---

## Repository Structure

```text
├── Pipline
│   ├── ExampleData
│   │   ├── S001/
│   │   ├── S002/
│   │   └── S003/
│   ├── MIClassification
│   │   ├── MIRest
│   │   │   ├── models/
│   │   │   ├── Train_MI_Rest_from_fif.py
│   │   │   └── Train_Simulate_MI_Rest_bandpower_vs_csp.py
│   │   ├── MIRestTypes
│   │   │   ├── Sanity_Check_2Stages.py
│   │   │   └── tryout_S001_two_stages.py
│   │   └── MITypes
│   │       ├── models/
│   │       ├── Train_MI_Types_from_fif.py
│   │       └── Train_Simulate_MI_Types_csp.py
│   └── Training_Data_Acquisition
│       ├── lsl_record_to_fif.py
│       ├── mi_cues_markers.py
│       └── recordings/
├── README.md
```

---

## Training Data Source (Pipeline A)

The initial models and simulation scripts were developed and validated using a publicly available benchmark dataset:

**PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)**

```text
https://www.physionet.org/content/eegmmidb/1.0.0/
```

This dataset contains motor imagery and motor movement recordings (including fists and feet) collected under controlled laboratory conditions. Performance obtained on this dataset is **not a guarantee** for stage conditions; final tuning must be done with your own recordings.

---

## Concept: Two-Stage MI Classifier

### Stage 1 – MI vs Rest

**Goal:** Detect whether the subject is performing Motor Imagery.

- Input: continuous EEG
- Output: probability `p(MI)`
- Methods used in this repo: **CSP + LDA** (and optional comparison vs bandpower-based baseline in the PhysioNet script)

### Stage 2 – MI Type (Fists vs Feet)

**Goal:** If MI is detected, classify which imagery type is performed.

- Input: EEG windows during MI
- Output: probability `p(feet)` (class 1 = feet, class 0 = fists)
- Method used in this repo: **CSP + LDA**

### Cascade usage

```text
EEG → Stage 1 (MI vs Rest)
        ↓ if MI detected (gate)
        Stage 2 (Fists vs Feet)
```

---

## Pipeline A: PhysioNet-Based Experiments

These scripts are primarily for algorithm development and benchmarking.

### Stage 1 (MI vs Rest) – PhysioNet

- Script: `Pipline/MIClassification/MIRest/Train_Simulate_MI_Rest_bandpower_vs_csp.py`
- Purpose:
  - load EDF runs
  - filter
  - train MI vs Rest
  - validate (leave-one-run-out)
  - simulate sliding-window probability stream
  - save models

### Stage 2 (Fists vs Feet) – PhysioNet

- Script: `Pipline/MIClassification/MITypes/Train_Simulate_MI_Types_csp.py`
- Purpose:
  - train T1 (fists) vs T2 (feet)
  - validate (leave-one-run-out)
  - save model

> Note: These scripts assume PhysioNet-style EDF naming (`S###R##.edf`) and that you place data under `Pipline/ExampleData/S00X/`.

---

## Pipeline B: Own Data (LSL → FIF → Training)

Pipeline B is designed for collecting and training on **your own recordings** (stage-relevant conditions).

### B1. Cue Presentation + Marker Stream

- Script: `Pipline/Training_Data_Acquisition/mi_cues_markers.py`
- What it does:
  - presents visual cues (colors; optional text)
  - broadcasts markers via LSL
  - labels:
    - `T0` = Rest
    - `T1` = Fists MI
    - `T2` = Feet MI
- Controls:
  - `ESC` → stop immediately
  - `SPACE` → skip cue (marker alignment preserved)

### B2. Record EEG + Markers to FIF

- Script: `Pipline/Training_Data_Acquisition/lsl_record_to_fif.py`
- What it does:
  - records EEG LSL stream + marker LSL stream
  - saves a time-locked `*_raw.fif` with MNE annotations
- Output directory:
  - `Pipline/Training_Data_Acquisition/recordings/`

### B3. Train Stage 1 + Stage 2 on Own Recordings

- Stage 1: `Pipline/MIClassification/MIRest/Train_MI_Rest_from_fif.py`
- Stage 2: `Pipline/MIClassification/MITypes/Train_MI_Types_from_fif.py`

Both scripts:
- load the recorded FIF
- extract windows from T0/T1/T2
- train CSP + LDA
- save joblib models into their respective `models/` folders

### B4. Sanity Check (Pseudo-Online)

- Script: `Pipline/MIClassification/MIRestTypes/Sanity_Check_2Stages.py`
- What it does:
  - loads Stage 1 + Stage 2 joblib models
  - slides windows through the recording
  - computes `p(MI)` and gated `p(feet)`
  - plots outputs aligned with annotations

This is the recommended verification step before integrating into TouchDesigner.

---

## Quickstart (Own Data)

Run the cue script and recorder in parallel, then train and sanity-check.

### 1) Cue presentation (Terminal A)

```bash
python Pipline/Training_Data_Acquisition/mi_cues_markers.py
```

### 2) Record EEG + markers (Terminal B)

```bash
python Pipline/Training_Data_Acquisition/lsl_record_to_fif.py
```

### 3) Train Stage 1 (MI vs Rest)

```bash
python Pipline/MIClassification/MIRest/Train_MI_Rest_from_fif.py
```

### 4) Train Stage 2 (Fists vs Feet)

```bash
python Pipline/MIClassification/MITypes/Train_MI_Types_from_fif.py
```

### 5) Sanity check the full cascade

```bash
python Pipline/MIClassification/MIRestTypes/Sanity_Check_2Stages.py
```

---

## Channel Selection (CHANNEL_GROUP)

Stage 1 and Stage 2 training scripts support a single knob:

```python
CHANNEL_GROUP = "all"          # use all EEG channels
# or
CHANNEL_GROUP = "distributed16"  # distributed subset
# or
CHANNEL_GROUP = "motor_roi"      # motor cortex ROI (e.g. C3/Cz/C4 neighborhood)
```

If channels are properly named (C3/Cz/C4 etc.), selection is name-based; otherwise the scripts fall back to index-based picks until a device-specific channel map is provided.

**Important:** Keep channel selection consistent between training and online deployment.

---

## Parameters to Revalidate on Real Device Data

When switching from PhysioNet to real stage recordings, the following parameters often need retuning:

- frequency band (`FREQ_BAND`)
- epoch timing (`TMIN`, `TMAX`)
- CSP components (`CSP_COMPONENTS`)
- online windowing (`WIN_LEN`, `WIN_STEP`)
- Stage-1 gate settings (threshold/dwell/refractory; used in sanity checks / later deployment)

---

## Troubleshooting

- LSL streams must be active and discoverable before recording.
- On macOS, if `pylsl` cannot find `liblsl`, install via conda-forge (`liblsl`) or Homebrew.
- You can stop recordings early; training just needs enough examples:
  - Stage 1 needs T0 and (T1/T2)
  - Stage 2 needs both T1 and T2

---

## Current Project State

- Stage 1 and Stage 2 training scripts implemented
- PhysioNet simulation scripts implemented
- Two-stage pseudo-online sanity check implemented
- Models exported as joblib for later deployment (e.g. TouchDesigner)

Next steps:
- validate with real device data
- tune thresholds / dwell for reliable stage behavior
- integrate real-time inference into TouchDesigner and map outputs to visuals/audio