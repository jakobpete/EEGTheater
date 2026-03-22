"""
live_lsl_feature_head.py

Ziel dieses Skriptkopfteils:
- Einen EEG-Datenstream über LSL (Lab Streaming Layer) finden
- Den Stream öffnen und Metadaten auslesen (Samplingrate, Kanalanzahl, Kanalnamen)
- Motor-relevante Kanäle (C3, C4, CPZ) identifizieren (falls vorhanden)
- Einen Ringbuffer (zirkulären Puffer) initialisieren, der laufend die neuesten Samples hält
- Hilfsfunktionen bereitstellen, um:
    (a) neue Samples in den Ringbuffer zu schreiben
    (b) das letzte Analysefenster (Sliding Window) als Array herauszuschneiden

Wichtig:
- Das ersetzt deinen bisherigen Offline-Teil (MNE read_raw_edf + raw.get_data).
- Im Live-Fall gibt es kein "data, times = ..."; die Daten kommen kontinuierlich.
"""

import time
import numpy as np
from pylsl import StreamInlet, resolve_stream


# ============================================================
# 1) LSL Stream finden + Inlet öffnen
# ============================================================

# LSL "type" ist ein Metadatum, das der Sender setzt.
# Bei EEG-Streams ist das sehr häufig "EEG".
LSL_TYPE = "EEG"

# Optional: Falls du den Streamnamen kennst (aus dem LSL Browser),
# kannst du ihn hier eintragen, um präzise genau diesen Stream zu wählen.
# Wenn None, wird der erste Stream mit type="EEG" verwendet.
LSL_NAME = None  # z.B. "RBE-16-X2" oder exakter Name aus der Hersteller-Software

# Wie lange wir maximal suchen, bevor wir abbrechen.
TIMEOUT_RESOLVE = 5  # Sekunden

print("Suche LSL-Stream...")

# resolve_stream fragt das LSL-Netzwerk nach Streams, die Kriterien erfüllen.
# Zwei typische Strategien:
# - "type" filtern (robust, wenn type sauber gesetzt ist)
# - "name" filtern (präziser, wenn Name eindeutig ist)
if LSL_NAME:
    # Suche nach Stream mit exaktem Namen
    streams = resolve_stream("name", LSL_NAME, timeout=TIMEOUT_RESOLVE)
else:
    # Suche nach Streams, deren 'type' dem gewünschten entspricht
    streams = resolve_stream("type", LSL_TYPE, timeout=TIMEOUT_RESOLVE)

# Wenn kein Stream gefunden wird, ist entweder:
# - die Sender-Software nicht gestartet
# - type/name stimmen nicht
# - LSL nicht im gleichen Netzwerk / gleiche Maschine erreichbar
if not streams:
    raise RuntimeError(
        "Kein passender LSL-Stream gefunden. "
        "Prüfe: läuft der Sender, stimmen type/name, ist LSL erreichbar?"
    )

# StreamInlet ist die "Empfangsseite" für einen LSL Stream.
# max_buflen: interner Puffer in Sekunden, den pylsl hält (nicht unser Ringbuffer!).
# Das hilft, kurzfristige Verzögerungen zu überbrücken.
inlet = StreamInlet(streams[0], max_buflen=60)

# info enthält Stream-Metadaten (Name, Type, Kanalanzahl, nominale Samplingrate etc.)
info = inlet.info()


# ============================================================
# 2) Stream-Metadaten auslesen (sfreq, Kanalnamen)
# ============================================================

# nominal_srate ist die vom Sender deklarierte Samplingrate.
# Achtung: manche Streams setzen das auf 0 (irregular sampling).
# Für Bandpower brauchst du aber eine sinnvolle sfreq.
sfreq = float(info.nominal_srate())

# Anzahl Kanäle (z.B. 16, 32 ...)
n_channels = int(info.channel_count())

print(f"LSL Stream gefunden: name='{info.name()}', type='{info.type()}'")
print(f"Kanalanzahl: {n_channels}")
print(f"Nominale Samplingrate: {sfreq} Hz")

# Kanalnamen:
# LSL Streams können (optional) Kanalmetadaten als XML in info.desc() bereitstellen.
# Viele EEG-Sender liefern so labels wie "C3", "C4", "PZ" etc.
#
# Falls der Sender keine Labels liefert, bauen wir fallback-Namen: CH1, CH2, ...
ch_names = []
try:
    # Navigiere im XML: <desc><channels><channel>...</channel></channels></desc>
    ch = info.desc().child("channels").child("channel")
    for _ in range(n_channels):
        label = ch.child_value("label")  # kann leer sein
        ch_names.append(label if label else f"CH{len(ch_names)+1}")
        ch = ch.next_sibling()
except Exception:
    # Wenn XML-Struktur fehlt oder Fehler beim Parsen auftreten:
    ch_names = [f"CH{i+1}" for i in range(n_channels)]

# Wie in deinem Offline-Code: Kanalnamen vereinheitlichen (Punkte entfernen, uppercase)
# Damit z.B. "C3." -> "C3" wird und du zuverlässig suchen kannst.
ch_names = [name.replace(".", "").upper() for name in ch_names]

print(f"Kanalnamen: {ch_names}")


# ============================================================
# 3) Motor-relevante Kanäle (C3/C4/CPZ) identifizieren
# ============================================================

# Wir bauen ein Dictionary MOTOR_CH, das z.B. sagt:
# MOTOR_CH["C3"] = Index im Datenarray
# Das ist nützlich, um später gezielt C3/C4 zu extrahieren.
MOTOR_CH = {}
for name in ["C3", "C4", "CPZ"]:
    if name in ch_names:
        MOTOR_CH[name] = ch_names.index(name)

print("Motor-Kanäle gefunden:", MOTOR_CH)


# ============================================================
# 4) Fenster-/Schritt-/Buffer-Parameter festlegen
# ============================================================

# WINDOW_SEC: Länge eines Analysefensters (Sliding Window) in Sekunden.
# STEP_SEC:   wie oft du ein neues Feature berechnen willst (Update-Rate).
# BUFFER_SEC: wie viele Sekunden du im Ringbuffer speicherst.
#
# Idee:
# - Der Ringbuffer hält immer die letzten BUFFER_SEC Sekunden.
# - Für Feature-Berechnung schneiden wir daraus das letzte WINDOW_SEC Fenster.
WINDOW_SEC = 2.0
STEP_SEC = 0.5
BUFFER_SEC = 20.0

# Um in Samples zu rechnen, multiplizieren wir mit sfreq:
# Beispiel: sfreq=250 Hz -> 2s = 500 Samples
#
# Hinweis: Wenn sfreq==0, musst du hier anders vorgehen (timestamp-basiert).
window_samples = int(WINDOW_SEC * sfreq) if sfreq > 0 else None
step_samples   = int(STEP_SEC * sfreq) if sfreq > 0 else None
buffer_samples = int(BUFFER_SEC * sfreq) if sfreq > 0 else None

if sfreq <= 0:
    raise RuntimeError(
        "Der Stream meldet nominal_srate=0. "
        "Für Bandpower brauchst du eine feste Samplingrate. "
        "Prüfe Vendor-Settings oder implementiere timestamp-basierte Resampling-Logik."
    )

print(
    f"WINDOW: {WINDOW_SEC}s ({window_samples} samples) | "
    f"STEP: {STEP_SEC}s ({step_samples} samples) | "
    f"BUFFER: {BUFFER_SEC}s ({buffer_samples} samples)"
)


# ============================================================
# 5) Ringbuffer (zirkulärer Puffer) initialisieren
# ============================================================

# Der Ringbuffer ist ein 2D-Array:
# shape = (n_channels, buffer_samples)
#
# Warum Ringbuffer?
# - Live kommt ständig neue Daten rein
# - Du willst aber immer nur die "letzten N Sekunden" behalten
# - Mit einem Ringbuffer überschreibst du zyklisch alte Daten.
buffer = np.zeros((n_channels, buffer_samples), dtype=np.float32)

# write_pos ist der Index (Spalte), wo das nächste Sample reingeschrieben wird.
write_pos = 0

# filled sagt, wie viele Samples schon gültig sind.
# Am Anfang ist der Buffer noch nicht voll, dann kann man noch kein volles Window schneiden.
filled = 0


# ============================================================
# 6) Hilfsfunktionen: Samples schreiben + letztes Fenster holen
# ============================================================

def push_samples(samples_2d: np.ndarray) -> None:
    """
    Schreibt neue Samples in den Ringbuffer.

    Parameter
    ---------
    samples_2d : np.ndarray
        Erwartete Form: (n_samples, n_channels)

        Genau so liefert pylsl typischerweise pull_chunk:
            samples = [[ch1, ch2, ... chN],   # Sample 1
                       [ch1, ch2, ... chN],   # Sample 2
                       ...]
        Wenn du daraus np.asarray machst, ist die Form (n_samples, n_channels).

    Wirkung
    -------
    - Transponiert die Daten zu (n_channels, n_samples), weil unser Buffer so strukturiert ist.
    - Schreibt sampleweise zyklisch in buffer[:, write_pos].
    - Aktualisiert write_pos und filled.
    """
    global buffer, write_pos, filled

    if samples_2d.ndim != 2:
        raise ValueError(f"samples_2d muss 2D sein, ist aber {samples_2d.ndim}D")

    if samples_2d.shape[1] != n_channels:
        raise ValueError(
            f"Unerwartete Kanalanzahl: samples haben {samples_2d.shape[1]} Kanäle, "
            f"Stream erwartet {n_channels}"
        )

    # Transponieren: (n_samples, n_channels) -> (n_channels, n_samples)
    x = samples_2d.T
    n_new = x.shape[1]

    # Wir schreiben sampleweise in den Ringbuffer.
    # (Das ist simpel und robust; später kann man das auch vektorisieren.)
    for i in range(n_new):
        buffer[:, write_pos] = x[:, i]
        write_pos = (write_pos + 1) % buffer.shape[1]
        filled = min(filled + 1, buffer.shape[1])


def get_latest_window(window_len_samples: int) -> np.ndarray | None:
    """
    Schneidet das letzte vollständige Fenster aus dem Ringbuffer heraus.

    Parameter
    ---------
    window_len_samples : int
        Länge des gewünschten Fensters in Samples (z.B. window_samples).

    Returns
    -------
    np.ndarray | None
        - Falls noch nicht genug Daten gesammelt wurden: None
        - Sonst: Array shape (n_channels, window_len_samples),
          das die zuletzt angekommenen Samples enthält.

    Erklärung
    ---------
    Ringbuffer ist zyklisch. Wenn der Schreibindex write_pos "umspringt",
    kann ein zusammenhängendes Fenster über das Buffer-Ende hinauslaufen.
    Dann müssen wir zwei Stücke concatenaten:
        buffer[:, start:] + buffer[:, :end]
    """
    if filled < window_len_samples:
        # Noch nicht genug Daten für ein vollständiges Fenster
        return None

    end = write_pos
    start = (end - window_len_samples) % buffer.shape[1]

    if start < end:
        # Normalfall: Fenster ist zusammenhängend im Buffer
        return buffer[:, start:end]
    else:
        # Wrap-around Fall: Fenster splitten und wieder zusammensetzen
        return np.concatenate([buffer[:, start:], buffer[:, :end]], axis=1)


# ============================================================
# 7) Beispiel: Live-Loop zum Befüllen + Fenster schneiden
# ============================================================

# Dieser Loop ist ein "Skeleton":
# - er zeigt dir, wie pull_chunk -> push_samples -> get_latest_window zusammenhängen
# - hier würdest du später deine compute_bandpower / Feature-Logik einsetzen

print("Starte Live-Loop (Skeleton). Abbrechen mit Ctrl+C.")

sample_counter = 0  # zählt, wie viele neue Samples seit letzter Feature-Berechnung angekommen sind

try:
    while True:
        # pull_chunk holt mehrere Samples auf einmal (Batch),
        # damit das effizienter ist als Sample-by-sample.
        #
        # timeout=0.0: non-blocking (wir warten nicht)
        # max_samples: begrenzt, wie viele Samples wir pro Aufruf maximal holen
        samples, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=step_samples)

        if samples:
            # Konvertiere in numpy; dtype float32 reicht für EEG und ist schneller/kleiner
            samples = np.asarray(samples, dtype=np.float32)
            push_samples(samples)
            sample_counter += samples.shape[0]

        # Sobald ungefähr step_samples neue Samples da sind, berechnen wir ein neues Fenster
        if sample_counter >= step_samples:
            sample_counter = 0

            window_data = get_latest_window(window_samples)
            if window_data is None:
                # noch nicht genug Daten
                continue

            # Hier würdest du jetzt Feature-Berechnung machen:
            # theta = compute_bandpower(window_data, 4, 8, sfreq) etc.
            #
            # Fürs Debuggen: nur Shape anzeigen
            print(f"Window ready: shape={window_data.shape}")

        # Kleine Pause verhindert 100% CPU, wenn gerade keine Samples kommen.
        time.sleep(0.001)

except KeyboardInterrupt:
    print("Live-Loop beendet.")
