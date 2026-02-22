# -------- LSL Receiver + 2-Stage MI Inference (CSP+LDA) ONLY --------
# TouchDesigner Script CHOP Callbacks DAT
#
# Outputs:
#   /<stream>/<type>/raw/ch{i}         (ALL raw channels, optional bandpass + optional CAR)
#   /<stream>/<type>/lsl_ts
#   /<stream>/<type>/mi/p_mi
#   /<stream>/<type>/mi/gate_open
#   /<stream>/<type>/mi/p_feet
#   status
#
# Notes:
# - NO bandpower computation at all.
# - Pulls ALL available LSL samples each cook.
# - MI channel selection:
#     If MI/Usemodelsubset = True -> uses cfg1['channel_group'] ('all','distributed16','motor_roi')
#     else -> uses MI/Channames + MI/Chanfallback (manual).
#
# TD Python packages needed:
#   pylsl, numpy, scipy, joblib, sklearn, mne
#
from pylsl import StreamInlet, resolve_streams
import time
import numpy as np

# ---------------- Channel subsets (match training scripts) ----------------
DISTRIBUTED_16_NAMES = [
    "Fp1", "Fp2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T7", "T8",
    "P7", "P8",
]

MOTOR_ROI_NAMES = [
    "FC3", "FCz", "FC4",
    "C3", "Cz", "C4",
    "CP3", "CPz", "CP4",
]

DISTRIBUTED_16_INDEX_FALLBACK = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
MOTOR_ROI_INDEX_FALLBACK = [6, 7, 8, 14, 15, 16, 22, 23, 24]

CHANNEL_GROUPS = {
    "all": {"names": None, "idx_fallback": None},
    "distributed16": {"names": DISTRIBUTED_16_NAMES, "idx_fallback": DISTRIBUTED_16_INDEX_FALLBACK},
    "motor_roi": {"names": MOTOR_ROI_NAMES, "idx_fallback": MOTOR_ROI_INDEX_FALLBACK},
}

# ---------------- persistent globals ----------------
if "lsl_inlet" not in globals(): lsl_inlet = None
if "n_channels" not in globals(): n_channels = 0
if "last_sample" not in globals(): last_sample = []
if "last_ts" not in globals(): last_ts = float("nan")
if "last_log_t" not in globals(): last_log_t = 0.0
if "last_resolve_t" not in globals(): last_resolve_t = 0.0
if "resolve_interval" not in globals(): resolve_interval = 2.0
if "current_stream_name" not in globals(): current_stream_name = ""
if "current_stream_type" not in globals(): current_stream_type = ""
if "lsl_labels" not in globals(): lsl_labels = []

# optional raw bandpass filter state (biquad cascade)
if "bp_z1" not in globals(): bp_z1 = None
if "bp_z2" not in globals(): bp_z2 = None
if "bp_sos" not in globals(): bp_sos = None
if "bp_cached" not in globals(): bp_cached = {}

# ---------------- MI globals ----------------
if "stage1" not in globals(): stage1 = None
if "stage2" not in globals(): stage2 = None
if "cfg1" not in globals(): cfg1 = {}
if "cfg2" not in globals(): cfg2 = {}

if "mi_pick_idx" not in globals(): mi_pick_idx = None
if "mi_pick_names" not in globals(): mi_pick_names = []
if "mi_buf" not in globals(): mi_buf = None
if "mi_idx" not in globals(): mi_idx = 0
if "mi_fill" not in globals(): mi_fill = 0
if "mi_cached" not in globals(): mi_cached = {}
if "last_mi_compute_t" not in globals(): last_mi_compute_t = 0.0

if "p_mi_raw" not in globals(): p_mi_raw = 0.0
if "p_feet_raw" not in globals(): p_feet_raw = 0.5
if "p_mi_s" not in globals(): p_mi_s = 0.0
if "p_feet_s" not in globals(): p_feet_s = 0.5
if "gate_open" not in globals(): gate_open = 0.0

if "gate_above_run" not in globals(): gate_above_run = 0
if "gate_refr_count" not in globals(): gate_refr_count = 0
if "gate_latched" not in globals(): gate_latched = False


# ============================ TD PARAMETERS ============================
def onSetupParameters(scriptOp):
    # LSL
    p = scriptOp.appendCustomPage("LSL")
    p.appendStr("Streamname", label="Stream Name")
    p.appendStr("Streamtype", label="Stream Type")
    p.appendToggle("Active", label="Active")
    p.appendFloat("Samplerate", label="CHOP Cook Rate (Hz)")
    p.appendToggle("Debuglogs", label="Debug Logs")
    p.appendPulse("Reconnect")

    # Raw preprocessing
    f = scriptOp.appendCustomPage("Raw")
    f.appendToggle("Bpenable", label="Bandpass Raw (biquad)")
    f.appendFloat("Bplow", label="Bandpass Low (Hz)")
    f.appendFloat("Bphigh", label="Bandpass High (Hz)")
    f.appendInt("Bporder", label="Bandpass Order (even)")

    r = scriptOp.appendCustomPage("Reference")
    r.appendToggle("Avgreference", label="Average Reference (CAR)")
    r.appendToggle("Refraw", label="Apply CAR to Raw Output")
    r.appendToggle("Refmi", label="Apply CAR to MI Input")

    # Models
    m = scriptOp.appendCustomPage("Models")
    m.appendStr("Stage1path", label="Stage1 Joblib Path (MI vs REST)")
    m.appendStr("Stage2path", label="Stage2 Joblib Path (FISTS vs FEET)")
    m.appendPulse("Loadmodels", label="Load Models")

    # MI
    mi = scriptOp.appendCustomPage("MI")
    mi.appendToggle("Mienable", label="Enable MI Inference")
    mi.appendToggle("Usemodelsubset", label="Use Channel Subset From Model Config")
    mi.appendStr("Channames", label="Manual MI Channel Names (space-separated)")
    mi.appendStr("Chanfallback", label="Manual Fallback Indices (space-separated)")
    mi.appendFloat("Miwin", label="MI Window (s)")
    mi.appendFloat("Mistep", label="MI Step (s)")
    mi.appendFloat("Mithr", label="MI Threshold (0-1)")
    mi.appendFloat("Midwell", label="Dwell (s)")
    mi.appendFloat("Mirefr", label="Refractory (s)")
    mi.appendFloat("Mismooth", label="Smoothing alpha (0-1)")
    mi.appendToggle("Miverbose", label="MI Debug Prints")

    # defaults
    scriptOp.par.Streamname.val = "*"
    scriptOp.par.Streamtype.val = "EEG"
    scriptOp.par.Active.val = True
    scriptOp.par.Debuglogs.val = True
    scriptOp.par.Samplerate.val = 120.0

    scriptOp.par.Bpenable.val = False
    scriptOp.par.Bplow.val = 1.0
    scriptOp.par.Bphigh.val = 40.0
    scriptOp.par.Bporder.val = 4

    scriptOp.par.Avgreference.val = False
    scriptOp.par.Refraw.val = False
    scriptOp.par.Refmi.val = True

    scriptOp.par.Stage1path.val = ""
    scriptOp.par.Stage2path.val = ""

    scriptOp.par.Mienable.val = True
    scriptOp.par.Usemodelsubset.val = True
    scriptOp.par.Channames.val = "C3 Cz C4"
    scriptOp.par.Chanfallback.val = "0 1 2"
    scriptOp.par.Miwin.val = 2.0
    scriptOp.par.Mistep.val = 0.25
    scriptOp.par.Mithr.val = 0.75
    scriptOp.par.Midwell.val = 0.50
    scriptOp.par.Mirefr.val = 1.50
    scriptOp.par.Mismooth.val = 0.20
    scriptOp.par.Miverbose.val = False
    return


def onPulse(par):
    if par.name == "Reconnect":
        _reset_inlet()
    elif par.name == "Loadmodels":
        _load_models(par.owner)
    return


# ============================ MAIN COOK ============================
def onCook(scriptOp):
    global last_ts, last_sample, n_channels, last_log_t

    scriptOp.clear()

    rate = max(1.0, float(scriptOp.par.Samplerate.eval() or 60.0))
    scriptOp.rate = rate

    if not bool(scriptOp.par.Active):
        _write_outputs(scriptOp)
        return

    stream_name = str(scriptOp.par.Streamname.eval()).strip()
    stream_type = str(scriptOp.par.Streamtype.eval()).strip()
    debug = bool(scriptOp.par.Debuglogs)

    if not _ensure_inlet_resilient(stream_name, stream_type, debug):
        scriptOp.numSamples = 1
        scriptOp.appendChan("status")[0] = -1.0
        _write_outputs(scriptOp)
        return

    # sfreq
    try:
        sfreq = float(lsl_inlet.info().nominal_srate() or 0.0)
    except Exception:
        sfreq = 0.0
    if sfreq <= 0:
        sfreq = rate

    # pull ALL queued samples
    samples = []
    tss = []
    try:
        while True:
            s, ts = lsl_inlet.pull_sample(timeout=0.0)
            if s is None:
                break
            samples.append(s)
            tss.append(ts)
    except Exception:
        samples, tss = [], []

    # process samples
    for sample, ts in zip(samples, tss):
        raw = np.array(sample, dtype=np.float64)
        last_ts = float(ts)
        n_channels = max(n_channels, len(raw))

        # optional bandpass
        if bool(scriptOp.par.Bpenable):
            raw = _bandpass_sample(
                raw, sfreq,
                low=float(scriptOp.par.Bplow.eval() or 1.0),
                high=float(scriptOp.par.Bphigh.eval() or 40.0),
                order=int(scriptOp.par.Bporder.eval() or 4)
            )

        # optional CAR
        raw_for_output = raw
        raw_for_mi = raw

        if bool(scriptOp.par.Avgreference):
            mean_ref = np.nanmean(raw)
            if np.isfinite(mean_ref):
                if bool(scriptOp.par.Refraw):
                    raw_for_output = raw_for_output.copy()
                    raw_for_output[:] = raw_for_output[:] - mean_ref
                if bool(scriptOp.par.Refmi):
                    raw_for_mi = raw_for_mi.copy()
                    raw_for_mi[:] = raw_for_mi[:] - mean_ref

        last_sample = raw_for_output.tolist()

        # MI buffer update (uses subset)
        _push_mi_sample(raw_for_mi, scriptOp, sfreq)

    # MI inference (cadenced)
    _maybe_run_mi(scriptOp, sfreq)

    if debug and (time.time() - last_log_t) >= 5.0:
        preview = ", ".join(f"{v:.3f}" for v in last_sample[:min(8, len(last_sample))])
        print(f"[LSL] {time.strftime('%H:%M:%S')} ts={last_ts:.3f} vals=[{preview}]" + (" ..." if len(last_sample) > 8 else ""))
        last_log_t = time.time()

    _write_outputs(scriptOp)
    return


# ============================ OUTPUTS ============================
def _write_outputs(scriptOp):
    global n_channels, last_sample, last_ts, current_stream_name, current_stream_type
    global p_mi_s, p_feet_s, gate_open

    if n_channels == 0:
        n_channels = len(last_sample) if last_sample else 1

    scriptOp.numSamples = 1
    base_name = f"/{current_stream_name or 'unknown'}/{current_stream_type or 'data'}"

    # raw (all channels)
    for i in range(n_channels):
        val = last_sample[i] if i < len(last_sample) else float("nan")
        scriptOp.appendChan(f"{base_name}/raw/ch{i}")[0] = val

    # ts
    scriptOp.appendChan(f"{base_name}/lsl_ts")[0] = last_ts if last_ts == last_ts else float("nan")

    # MI
    scriptOp.appendChan(f"{base_name}/mi/p_mi")[0] = float(p_mi_s)
    scriptOp.appendChan(f"{base_name}/mi/gate_open")[0] = float(gate_open)
    scriptOp.appendChan(f"{base_name}/mi/p_feet")[0] = float(p_feet_s)

    scriptOp.appendChan("status")[0] = 1.0


# ============================ RESET ============================
def _reset_inlet():
    global lsl_inlet, n_channels, current_stream_name, current_stream_type, lsl_labels
    global bp_z1, bp_z2, bp_sos, bp_cached
    global stage1, stage2, cfg1, cfg2
    global mi_pick_idx, mi_pick_names, mi_buf, mi_idx, mi_fill, mi_cached
    global last_mi_compute_t, p_mi_raw, p_feet_raw, p_mi_s, p_feet_s, gate_open
    global gate_above_run, gate_refr_count, gate_latched

    lsl_inlet = None
    n_channels = 0
    current_stream_name = ""
    current_stream_type = ""
    lsl_labels = []

    bp_z1 = None
    bp_z2 = None
    bp_sos = None
    bp_cached = {}

    stage1 = None
    stage2 = None
    cfg1 = {}
    cfg2 = {}

    mi_pick_idx = None
    mi_pick_names = []
    mi_buf = None
    mi_idx = 0
    mi_fill = 0
    mi_cached = {}
    last_mi_compute_t = 0.0

    p_mi_raw = 0.0
    p_feet_raw = 0.5
    p_mi_s = 0.0
    p_feet_s = 0.5
    gate_open = 0.0

    gate_above_run = 0
    gate_refr_count = 0
    gate_latched = False


# ============================ LSL CONNECT ============================
def _read_lsl_labels(inlet):
    labels = []
    try:
        info = inlet.info()
        ch = info.desc().child("channels").child("channel")
        while ch.name() == "channel":
            lab = ch.child_value("label")
            if lab:
                labels.append(lab.strip())
            ch = ch.next_sibling()
    except Exception:
        pass
    return labels


def _ensure_inlet_resilient(stream_name, stream_type, debug=False):
    global lsl_inlet, n_channels, last_resolve_t, resolve_interval
    global current_stream_name, current_stream_type, lsl_labels

    if lsl_inlet is not None:
        return True

    now = time.time()
    if now - last_resolve_t < resolve_interval:
        return False
    last_resolve_t = now

    name = (stream_name or "").strip().lower()
    stype = (stream_type or "").strip().lower()

    try:
        streams = resolve_streams(0.2)
        if not streams:
            if debug:
                print("[LSL] No LSL streams visible.")
            return False

        def match(s):
            sname = (s.name() or "").strip().lower()
            styp = (s.type() or "").strip().lower()
            name_ok = (name in ("", "*")) or (sname == name)
            type_ok = (stype in ("", "*")) or (styp == stype)
            return name_ok and type_ok

        matches = [s for s in streams if match(s)]
        if not matches:
            if debug:
                print(f"[LSL] No match for name='{stream_name}' type='{stream_type}'")
            return False

        stream = matches[0]
        lsl_inlet = StreamInlet(stream)
        info = lsl_inlet.info()
        n_channels = info.channel_count() or 0
        current_stream_name = info.name() or "unnamed"
        current_stream_type = info.type() or "unknown"

        lsl_labels = _read_lsl_labels(lsl_inlet)

        if debug:
            try:
                srate = info.nominal_srate()
            except Exception:
                srate = None
            print(f"[LSL] Connected: {current_stream_name} ({n_channels} ch, type={current_stream_type}, srate={srate})")
            if lsl_labels:
                print("[LSL] labels:", lsl_labels[:min(16, len(lsl_labels))], ("..." if len(lsl_labels) > 16 else ""))
            else:
                print("[LSL] labels: (none) -> MI will use index fallbacks (or manual)")

        return True

    except Exception as e:
        if debug:
            print(f"[LSL] Error connecting: {e}")
        lsl_inlet = None
        n_channels = 0
        lsl_labels = []
        return False


# ============================ OPTIONAL RAW BANDPASS (BIQUAD) ============================
def _design_biquad(type_, fs, f0, Q):
    w0 = 2.0 * np.pi * (f0 / fs)
    cosw0 = np.cos(w0)
    sinw0 = np.sin(w0)
    alpha = sinw0 / (2.0 * Q)

    if type_ == "highpass":
        b0 = (1 + cosw0) / 2
        b1 = -(1 + cosw0)
        b2 = (1 + cosw0) / 2
        a0 = 1 + alpha
        a1 = -2 * cosw0
        a2 = 1 - alpha
    elif type_ == "lowpass":
        b0 = (1 - cosw0) / 2
        b1 = 1 - cosw0
        b2 = (1 - cosw0) / 2
        a0 = 1 + alpha
        a1 = -2 * cosw0
        a2 = 1 - alpha
    else:
        raise ValueError("unknown biquad type")

    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0
    return (b0, b1, b2, 1.0, a1, a2)


def _make_sos_bandpass(fs, low, high, order):
    order = int(order)
    if order < 2: order = 2
    if order % 2 != 0: order += 1
    sections_each = max(1, order // 2)

    low = max(0.001, float(low))
    high = max(low + 0.001, float(high))
    nyq = fs / 2.0
    low = min(low, nyq * 0.99)
    high = min(high, nyq * 0.99)
    Q = 0.70710678

    sos = []
    for _ in range(sections_each):
        sos.append(_design_biquad("highpass", fs, low, Q))
    for _ in range(sections_each):
        sos.append(_design_biquad("lowpass", fs, high, Q))
    return np.array(sos, dtype=np.float64)


def _bandpass_sample(x, fs, low, high, order):
    global bp_z1, bp_z2, bp_sos, bp_cached
    nch = len(x)
    key = (float(fs), float(low), float(high), int(order), nch)

    if bp_sos is None or bp_cached.get("key") != key:
        bp_sos = _make_sos_bandpass(fs, low, high, order)
        nsec = bp_sos.shape[0]
        bp_z1 = np.zeros((nch, nsec), dtype=np.float64)
        bp_z2 = np.zeros((nch, nsec), dtype=np.float64)
        bp_cached["key"] = key

    y = x.astype(np.float64, copy=True)
    for s in range(bp_sos.shape[0]):
        b0, b1, b2, a0, a1, a2 = bp_sos[s]
        z1 = bp_z1[:, s]
        z2 = bp_z2[:, s]

        out = b0 * y + z1
        z1_new = b1 * y - a1 * out + z2
        z2_new = b2 * y - a2 * out

        bp_z1[:, s] = z1_new
        bp_z2[:, s] = z2_new
        y = out

    return y


# ============================ MODEL CONFIG + PICKING ============================
def _cfg_get(cfg, *keys, default=None):
    for k in keys:
        if k in cfg:
            return cfg.get(k)
    return default


def _get_channel_group_from_cfg(cfg):
    cg = _cfg_get(cfg, "channel_group", "CHANNEL_GROUP", "channel_picks_desc", default="all")
    try:
        return str(cg).strip().lower()
    except Exception:
        return "all"


def _auto_resolve_picks_from_group(cfg, labels, n_total):
    global mi_pick_idx, mi_pick_names

    cg = _get_channel_group_from_cfg(cfg)
    if cg not in CHANNEL_GROUPS:
        cg = "all"

    group = CHANNEL_GROUPS[cg]
    want_names = group["names"]
    fb_idx = group["idx_fallback"]

    if want_names is None and fb_idx is None:
        mi_pick_idx = list(range(n_total))
        mi_pick_names = [labels[i] if (labels and i < len(labels)) else f"ch{i}" for i in mi_pick_idx]
        return cg

    if labels and want_names:
        low = [l.lower() for l in labels]
        idxs, used = [], []
        for nm in want_names:
            nm_low = nm.lower()
            if nm_low in low:
                i = low.index(nm_low)
                idxs.append(i)
                used.append(labels[i])
        if len(idxs) >= max(3, int(0.6 * len(want_names))):
            mi_pick_idx = idxs
            mi_pick_names = used
            return cg

    if fb_idx:
        mi_pick_idx = [i for i in fb_idx if 0 <= i < n_total]
        mi_pick_names = [labels[i] if (labels and i < len(labels)) else f"ch{i}" for i in mi_pick_idx]
        return cg

    mi_pick_idx = list(range(min(n_total, 8)))
    mi_pick_names = [labels[i] if (labels and i < len(labels)) else f"ch{i}" for i in mi_pick_idx]
    return cg


def _load_models(scriptOp):
    global stage1, stage2, cfg1, cfg2
    global mi_pick_idx, mi_pick_names, mi_cached
    try:
        import joblib
        p1 = joblib.load(str(scriptOp.par.Stage1path.eval()))
        p2 = joblib.load(str(scriptOp.par.Stage2path.eval()))
        stage1, cfg1 = p1["model"], (p1.get("config", {}) or {})
        stage2, cfg2 = p2["model"], (p2.get("config", {}) or {})

        mi_pick_idx = None
        mi_pick_names = []
        mi_cached.pop("n_total", None)

        print("[MI] Models loaded. channel_group =", _get_channel_group_from_cfg(cfg1))
    except Exception as e:
        stage1 = None
        stage2 = None
        cfg1 = {}
        cfg2 = {}
        print("[MI] ERROR loading models:", e)


# ============================ MI PREPROCESS + INFERENCE ============================
def _get_mi_window():
    global mi_buf, mi_idx
    return np.concatenate([mi_buf[:, mi_idx:], mi_buf[:, :mi_idx]], axis=1)


def _preprocess_for_cfg(x, fs, cfg):
    # x: (ch, time)
    from scipy.signal import butter, lfilter, iirnotch, resample_poly

    fb = _cfg_get(cfg, "freq_band", "FREQ_BAND", default=(8.0, 30.0))
    try:
        lo, hi = float(fb[0]), float(fb[1])
    except Exception:
        lo, hi = 8.0, 30.0

    notch = _cfg_get(cfg, "notch", "notch_hz", "NOTCH_HZ", default=None)
    rs = _cfg_get(cfg, "resample_sfreq", "RESAMPLE_SFREQ", default=None)

    y = x.astype(np.float64, copy=True)

    if notch is not None:
        try:
            nhz = float(notch)
            if nhz > 0:
                q = 30.0
                b, a = iirnotch(nhz / (fs / 2.0), q)
                y = lfilter(b, a, y, axis=-1)
        except Exception:
            pass

    nyq = fs / 2.0
    lo = max(0.1, min(lo, nyq * 0.99))
    hi = max(lo + 0.1, min(hi, nyq * 0.99))
    b, a = butter(4, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    y = lfilter(b, a, y, axis=-1)

    if rs is not None:
        try:
            fs_out = float(rs)
            if fs_out > 0 and abs(fs_out - fs) > 1e-6:
                up = int(round(fs_out))
                down = int(round(fs))
                y = resample_poly(y, up, down, axis=-1)
        except Exception:
            pass

    return y


def _maybe_run_mi(scriptOp, sfreq):
    global last_mi_compute_t
    global p_mi_raw, p_feet_raw, p_mi_s, p_feet_s, gate_open
    global gate_above_run, gate_refr_count, gate_latched
    global stage1, stage2, cfg1, cfg2
    global mi_buf, mi_fill

    if not bool(scriptOp.par.Mienable):
        return
    if stage1 is None or stage2 is None:
        return
    if mi_buf is None or mi_fill < mi_buf.shape[1]:
        return

    step_s = max(0.05, float(scriptOp.par.Mistep.eval() or 0.25))
    now = time.time()
    if (now - last_mi_compute_t) < step_s:
        return
    last_mi_compute_t = now

    x_win = _get_mi_window()

    # Stage 1
    try:
        x1 = _preprocess_for_cfg(x_win, sfreq, cfg1)
        p_mi_raw = float(stage1.predict_proba(x1[np.newaxis, :, :])[0, 1])
    except Exception:
        p_mi_raw = 0.0

    # Gate
    thr = float(scriptOp.par.Mithr.eval() or 0.75)
    dwell = float(scriptOp.par.Midwell.eval() or 0.5)
    refr = float(scriptOp.par.Mirefr.eval() or 1.5)

    dwell_n = max(1, int(np.ceil(dwell / step_s)))
    refr_n = int(np.ceil(refr / step_s))

    above = (p_mi_raw >= thr)

    if gate_refr_count > 0:
        gate_refr_count -= 1

    if above:
        gate_above_run += 1
    else:
        gate_above_run = 0
        gate_latched = False

    if (gate_above_run >= dwell_n) and (gate_refr_count == 0) and (not gate_latched):
        gate_latched = True
        gate_refr_count = refr_n

    gate_open = 1.0 if ((gate_above_run >= dwell_n) and above) else 0.0

    # Stage 2
    if gate_open > 0.5:
        try:
            x2 = _preprocess_for_cfg(x_win, sfreq, cfg2)
            p_feet_raw = float(stage2.predict_proba(x2[np.newaxis, :, :])[0, 1])
        except Exception:
            p_feet_raw = 0.5
    else:
        p_feet_raw = 0.5

    # Smoothing
    a = float(scriptOp.par.Mismooth.eval() or 0.2)
    a = min(max(a, 0.0), 1.0)
    p_mi_s = (1.0 - a) * p_mi_s + a * p_mi_raw
    p_feet_s = (1.0 - a) * p_feet_s + a * p_feet_raw


# ============================ END SCRIPT ============================