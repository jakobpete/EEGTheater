import numpy as np
import mne
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, cross_val_score
from pathlib import Path

# --- Paths ---
# Project layout assumed:
# EEGTheater/
#   Pipline/
#     classifier.py
#   Pipline/ExampleData/
#     S001R04.edf
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Pipline" / "ExampleData"

# --- Dataset selection ---
# EEGMMIDB motor imagery runs (per subject):
#   LR imagery (left vs right fist): R04, R08, R12
#   HF imagery (hands vs feet):      R06, R10, R14
SUBJECT = "S001"  # change to S002, ...

# Choose which two classes to decode.
# Set MODE = "LR" for left-vs-right (imagery runs: 4,8,12)
# Set MODE = "HF" for hands-vs-feet (imagery runs: 6,10,14)
MODE = "LR"  # "LR" or "HF"

RUNS_BY_MODE = {
    "LR": [4, 8, 12],
    "HF": [6, 10, 14],
}

run_list = RUNS_BY_MODE.get(MODE)
if run_list is None:
    raise ValueError("MODE must be 'LR' or 'HF'")

edf_paths = [DATA_DIR / f"{SUBJECT}R{r:02d}.edf" for r in run_list]
print("Using EDFs:")
for p in edf_paths:
    print(" -", p)

# Load + preprocess each run and extract epochs; then concatenate across runs.
all_epochs = []
all_groups = []  # run index per epoch for GroupKFold

for r, edf_path in zip(run_list, edf_paths):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Basic preprocessing
    raw.filter(7., 30., fir_design="firwin")  # mu+beta band
    raw.notch_filter([60.0])  # optional; dataset is US

    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw)
    print(f"event_id (R{r:02d}):", event_id)

    # Validate expected annotation codes exist
    required = ["T1", "T2"]
    missing = [k for k in required if k not in event_id]
    if missing:
        raise KeyError(
            f"Missing annotation codes {missing} in event_id={event_id} for {edf_path}. "
            f"Check that the EDF contains annotations."
        )

    # Map labels according to mode
    if MODE == "LR":
        class_map = {"left": event_id["T1"], "right": event_id["T2"]}
    else:  # MODE == "HF"
        class_map = {"hands": event_id["T1"], "feet": event_id["T2"]}

    # Epoch around task onset events only (T1/T2). Typical MI window: 0.5–4.0 s after cue.
    epochs = mne.Epochs(
        raw,
        events,
        event_id=class_map,
        tmin=0.5,
        tmax=4.0,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )

    # Store epochs + group labels (one group per run)
    all_epochs.append(epochs)
    all_groups.append(np.full(len(epochs), r, dtype=int))
    print(f"Run R{r:02d}: {len(epochs)} epochs")

# Concatenate across runs
epochs = mne.concatenate_epochs(all_epochs)
groups = np.concatenate(all_groups)
print("Total epochs:", len(epochs))

X = epochs.get_data()  # (n_epochs, n_channels, n_times)

# y labels: 0/1 for sklearn, stable across runs.
# Because we built epochs with event_id=class_map, the event codes are exactly the values in that dict.
# We'll infer mapping from the first run's class_map ordering by sorting unique codes.
unique_codes = np.unique(epochs.events[:, 2])
if len(unique_codes) != 2:
    raise ValueError(f"Expected 2 classes but got codes: {unique_codes}")
code_to_label = {code: i for i, code in enumerate(sorted(unique_codes))}
y = np.array([code_to_label[c] for c in epochs.events[:, 2]], dtype=int)
print("y counts:", np.bincount(y))

# 5) CSP + LDA classifier
csp = CSP(n_components=6, reg='ledoit_wolf', log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

clf = Pipeline([("csp", csp), ("lda", lda)])

# 6) CV (leave-one-run-out via GroupKFold)
cv = GroupKFold(n_splits=len(run_list))
scores = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="accuracy")

print("GroupCV accuracy: %.3f ± %.3f" % (scores.mean(), scores.std()))