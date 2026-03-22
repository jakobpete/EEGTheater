"""MI cue + marker sender for EEG training (LSL).

Purpose
-------
This script is used during data acquisition for motor-imagery (MI) classifiers.
It does two things in parallel:
  1) Shows visual cues to the actor (color screens, optionally with text overlay).
  2) Sends corresponding event markers over LSL and logs them to CSV.

Markers
-------
State markers are *state changes* (not time-locked trials):
  - T0 = REST
  - T1 = FISTS MI
  - T2 = FEET MI
The state is assumed to continue until the next marker arrives.

Protocol
--------
We run 3 blocks (B1/B2/B3). Block markers are emitted if enabled:
  - B1_START / B1_END
  - B2_START / B2_END
  - B3_START / B3_END

Controls
--------
  - SPACE: start (on READY) / skip to next cue immediately (during blocks)
  - ESC:   quit immediately (also during cue waiting)

Outputs
-------
  - <SUBJECT>_<SESSION_TAG>_<timestamp>_markerlog.csv
  - <SUBJECT>_<SESSION_TAG>_<timestamp>_marker_meta.json

Notes
-----
- The cue window is resizable. After resizing, the last screen is redrawn.
- LSL timestamps are written implicitly by LabStreamingLayer; CSV also logs local time.
"""

import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

# External dependencies: pygame (visual cues) and pylsl (LSL marker stream)
import pygame
from pylsl import StreamInfo, StreamOutlet

# -----------------------
# CONFIG
# -----------------------

# Session identity (used in filenames and LSL source_id)
SUBJECT = "Tryout_01"
SESSION_TAG = "MI_TRAINING"
OUTDIR = Path("Pipline/Training_Data_Acquisition/recordings")

# LSL stream identity for event markers (must match the recorder script)
LSL_MARKER_STREAM_NAME = "MI_Markers"
LSL_MARKER_STREAM_TYPE = "Markers"

# 3-block protocol structure (minutes per block)
BLOCK_MINUTES = [10, 10, 10]
USE_BLOCK_MARKERS = True

# Block 3 can be switched between presets without touching code logic
# Block 3 profile preset (change this depending on the artistic/technical goal)
#   - "frequent": MI appears often (good if you expect many triggers per minute on stage)
#   - "rare":     MI appears occasionally (good if MI is a special trigger)
BLOCK3_PRESET = "frequent"

BLOCK3_PROFILES = {
    # Frequent MI: similar balance to Blocks 1/2, but still short enough to practice quick onsets
    "frequent": {
        "rest_dur": (6, 15),
        "mi_dur": (3, 6),
        "probs": {"T0": 0.45, "T1": 0.275, "T2": 0.275},
        "intro": "BLOCK 3 (stage-like, frequent)\nViele kurze MI-Bursts.\nSPACE = continue",
    },
    # Rare MI: long rests, short MI bursts, dominated by REST
    "rare": {
        "rest_dur": (10, 20),
        "mi_dur": (3, 5),
        "probs": {"T0": 0.70, "T1": 0.15, "T2": 0.15},
        "intro": "BLOCK 3 (stage-like, rare)\nLange Ruhe, kurze MI-Bursts.\nSPACE = continue",
    },
}

TOTAL_MINUTES = sum([10, 10, 10])

# Default cue durations (seconds) for Blocks 1 & 2
REST_DUR_RANGE = (6, 15)   # seconds
MI_DUR_RANGE   = (6, 12)   # seconds

# Default state probabilities for Blocks 1 & 2 (must sum ~1.0)
P_REST, P_FISTS, P_FEET = 0.45, 0.275, 0.275

# Prevent staying in the same state too long (reduces boredom / habituation)
MAX_SAME_STATE_STREAK = 2

# Visual display settings
FONT_SIZE = 90
BG_COLOR = (0, 0, 0)
FG_COLOR = (240, 240, 240)


# Color-only cue mapping (minimizes language/semantic processing)
# Cue colors (RGB) instead of words
# Suggested mapping:
#   T0 (REST)  -> gray
#   T1 (FISTS) -> red
#   T2 (FEET)  -> blue

CUE_COLORS = {
    "T0": (120, 120, 120),  # gray
    "T1": (200, 40, 40),    # red
    "T2": (40, 80, 200),    # blue
}

# Debug option: overlay text label on top of the color cue
SHOW_STATE_TEXT = True

STATE_LABELS = {
    "T0": "REST",
    "T1": "FAUST",
    "T2": "FUSS",
}

# Utility: UTC timestamp string for human-readable logging
def now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"

def choose_next_state(prev, streak, just_had_mi, probs=None):
    """Choose the next state (T0/T1/T2) according to probabilities.

    - `probs` can be overridden per block.
    - After MI (T1/T2), we bias slightly toward REST (T0) to avoid rapid MI chaining.
    - `MAX_SAME_STATE_STREAK` prevents repeating the same state too often.
    """
    if probs is None:
        probs = {"T0": P_REST, "T1": P_FISTS, "T2": P_FEET}
    else:
        probs = dict(probs)  # copy

    if just_had_mi:
        probs["T0"] += 0.20
        probs["T1"] -= 0.10
        probs["T2"] -= 0.10

    if prev and streak >= MAX_SAME_STATE_STREAK:
        choices = [s for s in probs if s != prev]
        weights = [max(probs[s], 0.01) for s in choices]
        return random.choices(choices, weights=weights, k=1)[0]

    choices = list(probs.keys())
    weights = [max(probs[s], 0.01) for s in choices]
    return random.choices(choices, weights=weights, k=1)[0]

def dur_for(state, rest_range=REST_DUR_RANGE, mi_range=MI_DUR_RANGE):
    """Return a random duration for the current state.

    REST uses `rest_range`; MI states use `mi_range`.
    """
    if state == "T0":
        return random.randint(*rest_range)
    return random.randint(*mi_range)

def main():
    # --- Event-aware wait helper (keeps ESC/SPACE responsive during cue timing) ---
    def wait_interruptible(seconds: float) -> str:
        """Wait for `seconds` while still processing events.

        Returns:
          - "quit" if ESC was pressed
          - "skip" if SPACE was pressed (advance to next cue immediately)
          - "ok" if the wait completed normally
        """
        end_t = time.time() + float(seconds)
        while time.time() < end_t:
            for ev in pygame.event.get():
                handle_resize(ev)
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    return "quit"
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    return "skip"
            time.sleep(0.01)
        return "ok"

    # --- Output files for this session ---
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{SUBJECT}_{SESSION_TAG}_{ts}"
    events_path = OUTDIR / f"{base}_markerlog.csv"
    meta_path = OUTDIR / f"{base}_marker_meta.json"

    # --- LSL marker stream (single-channel string markers) ---
    info = StreamInfo(
        name=LSL_MARKER_STREAM_NAME,
        type=LSL_MARKER_STREAM_TYPE,
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id=f"{SUBJECT}_{ts}",
    )
    outlet = StreamOutlet(info)

    # --- Pygame window setup (resizable) ---
    pygame.init()
    WINDOW_SIZE = (900, 600)  # windowed for development
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("MI Training Cues")
    font = pygame.font.SysFont(None, FONT_SIZE)
    SMALL_FONT_SIZE = int(FONT_SIZE * 0.6)
    small_font = pygame.font.SysFont(None, SMALL_FONT_SIZE)

    current_view = "text"   # either "text" or "state"
    current_text = ""
    current_state = "T0"

    # Redraw logic: macOS/pygame can blank the window on resize, so we redraw last view
    def handle_resize(ev):
        nonlocal screen
        if ev.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(ev.size, pygame.RESIZABLE)
            # Redraw the last shown screen after resize
            if current_view == "state":
                draw_state(current_state)
            else:
                draw(current_text)

    # Draw a multi-line text screen (centered)
    def draw(msg):
        nonlocal current_view, current_text
        current_view = "text"
        current_text = msg
        screen.fill(BG_COLOR)
        lines = msg.split("\n")
        h = screen.get_height()
        for i, line in enumerate(lines):
            surf = font.render(line, True, FG_COLOR)
            rect = surf.get_rect(center=(screen.get_width() // 2, h // 2 + i * FONT_SIZE))
            screen.blit(surf, rect)
        pygame.display.flip()

    # Draw the current cue as a full-window color (optionally with a text overlay)
    def draw_state(state_key: str):
        nonlocal current_view, current_state
        current_view = "state"
        current_state = state_key
        # Full-window color cue
        color = CUE_COLORS.get(state_key, (0, 0, 0))
        screen.fill(color)

        # Optional text overlay
        if SHOW_STATE_TEXT:
            label = STATE_LABELS.get(state_key, state_key)
            surf = font.render(label, True, (255, 255, 255))
            rect = surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(surf, rect)

        pygame.display.flip()

    # Emit an LSL marker AND log it to CSV with local time + UTC string
    def emit_marker(marker: str):
        outlet.push_sample([marker])
        w.writerow([now_utc_iso(), time.time(), marker])

    # Session metadata (stored as JSON alongside the CSV)
    meta = {
        "subject": SUBJECT,
        "session_tag": SESSION_TAG,
        "start_utc": now_utc_iso(),
        "total_minutes": TOTAL_MINUTES,
        "durations": {"rest": REST_DUR_RANGE, "mi": MI_DUR_RANGE},
        "marker_stream": {"name": LSL_MARKER_STREAM_NAME, "type": LSL_MARKER_STREAM_TYPE},
        "notes": "Markers are state changes: state continues until next marker.",
        "blocks": {
            "minutes": BLOCK_MINUTES,
            "use_block_markers": USE_BLOCK_MARKERS,
            "block3": {
                "preset": BLOCK3_PRESET,
                "profiles": BLOCK3_PROFILES,
            },
        },
    }

    # --- CSV event log (one row per marker) ---
    with open(events_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utc_time", "local_time_s", "marker"])

        # READY screen: explains controls + cue mapping once; actual cues use only colors
        screen.fill(BG_COLOR)
        current_view = "text"
        current_text = "READY_SCREEN"

        center_x = screen.get_width() // 2
        center_y = screen.get_height() // 2

        # Large title
        title = font.render("READY", True, FG_COLOR)
        title_rect = title.get_rect(center=(center_x, center_y - FONT_SIZE * 2))
        screen.blit(title, title_rect)

        # Medium instruction line
        instr = small_font.render("SPACE = start/skip | ESC = quit", True, FG_COLOR)
        instr_rect = instr.get_rect(center=(center_x, center_y - FONT_SIZE))
        screen.blit(instr, instr_rect)

        # Smaller cue explanation
        lines = [
            "CUES:",
            "GRAU = REST (T0)",
            "ROT  = FAUST-MI (T1)",
            "BLAU = FUSS-MI  (T2)",
        ]
        for i, line in enumerate(lines):
            surf = small_font.render(line, True, FG_COLOR)
            rect = surf.get_rect(center=(center_x, center_y + i * SMALL_FONT_SIZE))
            screen.blit(surf, rect)

        pygame.display.flip()
        while True:
            for ev in pygame.event.get():
                handle_resize(ev)
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    break
            else:
                time.sleep(0.01)
                continue
            break

        # Fail early if the user configured an invalid Block 3 preset
        if BLOCK3_PRESET not in BLOCK3_PROFILES:
            raise ValueError(f"Invalid BLOCK3_PRESET='{BLOCK3_PRESET}'. Choose one of: {list(BLOCK3_PROFILES.keys())}")

        # Block specifications (timing/probabilities can differ per block)
        block_specs = [
            {
                "idx": 1,
                "minutes": BLOCK_MINUTES[0],
                "rest_range": REST_DUR_RANGE,
                "mi_range": MI_DUR_RANGE,
                "probs": {"T0": P_REST, "T1": P_FISTS, "T2": P_FEET},
                "intro": "BLOCK 1 (clean)\nBitte möglichst ruhig sitzen.\nSPACE = continue",
            },
            {
                "idx": 2,
                "minutes": BLOCK_MINUTES[1],
                "rest_range": REST_DUR_RANGE,
                "mi_range": MI_DUR_RANGE,
                "probs": {"T0": P_REST, "T1": P_FISTS, "T2": P_FEET},
                "intro": "BLOCK 2 (natural)\nKleine natürliche Bewegungen erlaubt.\nSPACE = continue",
            },
            {
                "idx": 3,
                "minutes": BLOCK_MINUTES[2],
                "rest_range": BLOCK3_PROFILES[BLOCK3_PRESET]["rest_dur"],
                "mi_range": BLOCK3_PROFILES[BLOCK3_PRESET]["mi_dur"],
                "probs": BLOCK3_PROFILES[BLOCK3_PRESET]["probs"],
                "intro": BLOCK3_PROFILES[BLOCK3_PRESET]["intro"],
            },
        ]

        # --- Run blocks sequentially ---
        for spec in block_specs:
            # Inter-block instruction screen (SPACE to continue)
            draw(spec["intro"])
            while True:
                for ev in pygame.event.get():
                    handle_resize(ev)
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                        break
                else:
                    time.sleep(0.01)
                    continue
                break

            bidx = spec["idx"]
            bstart_marker = f"B{bidx}_START"
            bend_marker = f"B{bidx}_END"

            # Optional block boundary markers (useful for later segmentation)
            if USE_BLOCK_MARKERS:
                emit_marker(bstart_marker)

            block_end = time.time() + spec["minutes"] * 60

            # Start each block in REST (T0)
            state = "T0"
            streak = 0
            just_had_mi = False

            emit_marker(state)
            draw_state(state)

            while time.time() < block_end:
                for ev in pygame.event.get():
                    handle_resize(ev)
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        meta["end_utc"] = now_utc_iso()
                        with open(meta_path, "w") as mf:
                            json.dump(meta, mf, indent=2)
                        pygame.quit()
                        return

                # Wait for the current cue duration (ESC quits, SPACE skips to next cue)
                wait_result = wait_interruptible(
                    dur_for(state, rest_range=spec["rest_range"], mi_range=spec["mi_range"])
                )
                if wait_result == "quit":
                    meta["end_utc"] = now_utc_iso()
                    with open(meta_path, "w") as mf:
                        json.dump(meta, mf, indent=2)
                    pygame.quit()
                    return

                # Choose and display the next state, then emit its marker immediately
                nxt = choose_next_state(state, streak, just_had_mi, probs=spec["probs"])
                streak = streak + 1 if nxt == state else 1
                state = nxt
                just_had_mi = state in ("T1", "T2")

                emit_marker(state)
                draw_state(state)

                # On SPACE-skip, continue immediately to allow fast stepping through cues
                if wait_result == "skip":
                    continue

            if USE_BLOCK_MARKERS:
                emit_marker(bend_marker)

        # End-of-session marker
        emit_marker("SESSION_END")
        draw("DONE\nESC = exit")

    meta["end_utc"] = now_utc_iso()
    with open(meta_path, "w") as mf:
        json.dump(meta, mf, indent=2)

    # Keep window open on DONE screen until ESC
    while True:
        for ev in pygame.event.get():
            handle_resize(ev)
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit()
                return
        time.sleep(0.05)

if __name__ == "__main__":
    main()