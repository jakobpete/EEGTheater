import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import mne
import matplotlib.pyplot as plt  # Requires matplotlib (conda/pip install matplotlib)
from pylsl import resolve_streams, StreamInlet

# -----------------------
# CONFIG
# -----------------------
# OUTDIR = Path("recordings")
OUTDIR = Path("Pipline/Training_Data_Acquisition/recordings")
SUBJECT = "Tryout_01"
SESSION_TAG = "MI_TRAINING"

# NEW: marker-only mode for testing (no EEG device needed)
RECORD_EEG = False  # <-- set True when the device arrives

# Live visualization (only used when RECORD_EEG=True)
LIVE_PLOT = True
PLOT_LAST_SECONDS = 10.0          # seconds of history to display
PLOT_UPDATE_HZ = 5.0              # refresh rate
PLOT_CHANNEL_INDICES = [0, 1]     # channel indices to plot (adjust later to match e.g. C3/C4)
PLOT_DOWNSAMPLE = 1               # >1 to downsample for faster plotting

# How long to record (seconds). Set None to record until you press Ctrl+C.
DURATION_S: Optional[int] = None

# Stream selection (adjust later if vendor uses other types/names)
EEG_TYPE_CANDIDATES = {"EEG"}          # some vendors use "EEG", others use e.g. "eeg" or "EEGStream"
MARKER_STREAM_NAME = "MI_Markers"      # must match your cue script
MARKER_TYPE = "Markers"

# Waiting behavior
WAIT_POLL_S = 1.0        # how often to re-check for streams
WAIT_TIMEOUT_S = None    # None = wait forever; or set e.g. 120 for 2 minutes

# Pulling behavior
PULL_CHUNK_SEC = 0.25  # how often to pull chunks

@dataclass
class StreamPick:
    name: str
    stype: str
    nchan: int
    srate: float

def utc_iso():
    return datetime.utcnow().isoformat() + "Z"

def find_stream(streams, want_type: Optional[str] = None, want_name: Optional[str] = None):
    """Return the first matching LSL stream info, or None."""
    for s in streams:
        if want_type is not None and s.type() != want_type:
            continue
        if want_name is not None and s.name() != want_name:
            continue
        return s
    return None

def wait_for_stream(want_type: Optional[str], want_name: Optional[str], *,
                    type_candidates: Optional[set] = None,
                    label: str = "stream"):
    """
    Wait until a stream appears. If type_candidates is provided, matches any of those types.
    """
    t0 = time.time()
    while True:
        streams = resolve_streams(wait_time=1.0)

        if type_candidates is not None:
            for s in streams:
                if s.type() in type_candidates and (want_name is None or s.name() == want_name):
                    return s

        s = find_stream(streams, want_type=want_type, want_name=want_name)
        if s is not None:
            return s

        if WAIT_TIMEOUT_S is not None and (time.time() - t0) > WAIT_TIMEOUT_S:
            raise TimeoutError(f"Timed out waiting for {label} (type={want_type}, name={want_name}).")

        print(f"Waiting for {label}... (type={want_type or type_candidates}, name={want_name})")
        time.sleep(WAIT_POLL_S)

def make_inlet(stream_info):
    return StreamInlet(stream_info, max_buflen=360, max_chunklen=0)

def describe_stream(stream_info) -> StreamPick:
    return StreamPick(
        name=stream_info.name(),
        stype=stream_info.type(),
        nchan=stream_info.channel_count(),
        srate=float(stream_info.nominal_srate()),
    )

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{SUBJECT}_{SESSION_TAG}_{ts}"

    fif_path = OUTDIR / f"{base}_raw.fif"
    events_path = OUTDIR / f"{base}_events.csv"
    meta_path = OUTDIR / f"{base}_meta.json"

    print("=== Python LSL Recorder ===")
    print(f"RECORD_EEG = {RECORD_EEG}")
    print("Press Ctrl+C to stop recording.\n")

    # Always wait for marker stream (your cue script may start after recorder)
    marker_info = wait_for_stream(
        want_type=MARKER_TYPE,
        want_name=MARKER_STREAM_NAME,
        label="marker stream"
    )
    marker_pick = describe_stream(marker_info)
    marker_inlet = make_inlet(marker_info)
    print(f"Marker stream found: {marker_pick}")

    eeg_pick = None
    eeg_inlet = None

    # Live plot variables (initialized for both modes)
    plot_enabled = False
    fig = ax = None
    lines = None
    buf = None
    plot_period = None
    next_plot_time = None

    # NEW: only wait for EEG if RECORD_EEG is True
    if RECORD_EEG:
        eeg_info = wait_for_stream(
            want_type=None,
            want_name=None,
            type_candidates=EEG_TYPE_CANDIDATES,
            label="EEG stream"
        )
        eeg_pick = describe_stream(eeg_info)
        eeg_inlet = make_inlet(eeg_info)
        print(f"EEG stream found: {eeg_pick}")
        # -----------------------
        # Live plot setup
        # -----------------------
        plot_enabled = bool(LIVE_PLOT)
        if plot_enabled:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_title(f"Live EEG (last {PLOT_LAST_SECONDS:.1f} s)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (a.u.)")

            n_hist = int(PLOT_LAST_SECONDS * eeg_pick.srate)
            if PLOT_DOWNSAMPLE > 1:
                n_hist = max(1, n_hist // PLOT_DOWNSAMPLE)

            buf = np.zeros((len(PLOT_CHANNEL_INDICES), n_hist), dtype=np.float32)
            tbuf = np.linspace(-PLOT_LAST_SECONDS, 0.0, n_hist, dtype=np.float32)

            lines = []
            for _ in range(len(PLOT_CHANNEL_INDICES)):
                (ln,) = ax.plot(tbuf, np.zeros_like(tbuf))
                lines.append(ln)

            ax.set_xlim(-PLOT_LAST_SECONDS, 0.0)
            fig.canvas.draw()
            fig.canvas.flush_events()

            plot_period = 1.0 / max(PLOT_UPDATE_HZ, 1e-6)
            next_plot_time = time.time() + plot_period
    else:
        print("Marker-only mode: not waiting for EEG stream.")

    # Storage
    eeg_samples: List[np.ndarray] = []
    eeg_ts: List[np.ndarray] = []
    markers: List[Tuple[float, str]] = []

    # Open events log
    with open(events_path, "w", newline="") as evfile:
        evw = csv.writer(evfile)
        evw.writerow(["lsl_time", "marker"])

        meta = {
            "subject": SUBJECT,
            "session_tag": SESSION_TAG,
            "start_utc": utc_iso(),
            "record_eeg": RECORD_EEG,
            "marker_stream": marker_pick.__dict__,
            "eeg_stream": eeg_pick.__dict__ if eeg_pick else None,
            "notes": "Markers are state changes: state continues until next marker.",
        }

        print("\nRecording...")
        t_start = time.time()

        try:
            while True:
                if DURATION_S is not None and (time.time() - t_start) > DURATION_S:
                    break

                # Pull EEG chunk if enabled
                if RECORD_EEG and eeg_inlet is not None and eeg_pick is not None:
                    max_samp = int(max(eeg_pick.srate * PULL_CHUNK_SEC, 1))
                    chunk, ts_chunk = eeg_inlet.pull_chunk(timeout=0.0, max_samples=max_samp)
                    if ts_chunk:
                        eeg_samples.append(np.asarray(chunk, dtype=np.float32))
                        eeg_ts.append(np.asarray(ts_chunk, dtype=np.float64))
                        # Update live plot buffer with newest samples
                        if plot_enabled and buf is not None and lines is not None:
                            arr = np.asarray(chunk, dtype=np.float32)  # (n_samples, n_channels)
                            if PLOT_DOWNSAMPLE > 1:
                                arr = arr[::PLOT_DOWNSAMPLE]

                            n_new = arr.shape[0]
                            if n_new >= buf.shape[1]:
                                arr = arr[-buf.shape[1]:]
                                n_new = arr.shape[0]
                                for i, ch_idx in enumerate(PLOT_CHANNEL_INDICES):
                                    buf[i, :] = arr[:, ch_idx]
                            else:
                                buf[:, :-n_new] = buf[:, n_new:]
                                for i, ch_idx in enumerate(PLOT_CHANNEL_INDICES):
                                    buf[i, -n_new:] = arr[:, ch_idx]

                            now = time.time()
                            if next_plot_time is not None and plot_period is not None and now >= next_plot_time:
                                y_min = float(np.min(buf))
                                y_max = float(np.max(buf))
                                if y_min == y_max:
                                    y_min -= 1.0
                                    y_max += 1.0
                                pad = 0.05 * (y_max - y_min)
                                ax.set_ylim(y_min - pad, y_max + pad)

                                for i, ln in enumerate(lines):
                                    ln.set_ydata(buf[i])

                                fig.canvas.draw_idle()
                                fig.canvas.flush_events()
                                next_plot_time = now + plot_period

                # Pull marker samples (can be multiple)
                while True:
                    m, t = marker_inlet.pull_sample(timeout=0.0)
                    if t is None:
                        break
                    marker = m[0] if isinstance(m, list) else str(m)
                    markers.append((float(t), str(marker)))
                    evw.writerow([t, marker])
                    print(f"[MARKER] {t:.3f} {marker}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopping...")

        meta["end_utc"] = utc_iso()

    # Always save meta + events
    with open(meta_path, "w") as mf:
        json.dump(meta, mf, indent=2)

    print(f"Saved: {events_path}")
    print(f"Saved: {meta_path}")

    # If marker-only, we are done
    if not RECORD_EEG:
        print("Done (marker-only). No FIF written.")
        return

    # Otherwise, build and save FIF
    if not eeg_samples:
        raise RuntimeError("RECORD_EEG=True but no EEG samples were recorded. Is the EEG stream running?")

    X = np.concatenate(eeg_samples, axis=0)  # (n_samples, n_channels)
    t_eeg = np.concatenate(eeg_ts, axis=0)   # (n_samples,)

    sfreq = float(eeg_pick.srate)
    ch_names = [f"EEG{idx+1:02d}" for idx in range(eeg_pick.nchan)]
    ch_types = ["eeg"] * eeg_pick.nchan
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(X.T, info)

    # Convert marker LSL timestamps to MNE Annotations (relative to first EEG sample)
    t0_lsl = float(t_eeg[0])
    if markers:
        onsets = [float(t - t0_lsl) for t, _ in markers]
        desc = [mk for _, mk in markers]
        raw.set_annotations(mne.Annotations(onset=onsets, duration=[0.0]*len(onsets), description=desc))

    raw.save(fif_path, overwrite=False)
    print(f"Saved: {fif_path}")
    print("Done.")

if __name__ == "__main__":
    main()