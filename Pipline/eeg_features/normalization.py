# normalization.py

import numpy as np


class BaselineNormalizer:
    """
    BaselineNormalizer

    Dieses Objekt übernimmt:
    1) Sammeln einer initialen Baseline über N Fenster
    2) Berechnung von mean und std
    3) Z-Score-Normalisierung neuer Werte
    4) Optional:
        - Clamping (z.B. auf [-3, +3])
        - exponentielle Glättung

    Typische Nutzung:
        bn = BaselineNormalizer(warmup_windows=40, clip=3.0, smooth_alpha=0.2)
        z = bn.update(x)   # -> None während Baseline, danach normalisierter Wert
    """

    def __init__(
        self,
        warmup_windows: int,
        clip: float | None = None,
        smooth_alpha: float | None = None,
        eps: float = 1e-15,
    ):
        # --- Konfiguration ---
        self.warmup_windows = int(warmup_windows)
        self.clip = clip                    # z.B. 3.0 oder None
        self.smooth_alpha = smooth_alpha    # z.B. 0.2 oder None
        self.eps = eps

        # --- interner Zustand ---
        self.values = []        # sammelt Baseline-Werte
        self.ready = False      # Baseline fertig?
        self.mean = None
        self.std = None
        self._smooth_val = None

    def update(self, x: float):
        """
        Übergibt einen neuen Feature-Wert x.

        Rückgabe:
        - None              → Baseline wird noch gesammelt
        - float (z-score)   → normalisierter Wert (optional geclippt & geglättet)
        """
        x = float(x)

        # -------------------------------------------------
        # 1) Baseline sammeln
        # -------------------------------------------------
        if not self.ready:
            self.values.append(x)

            if len(self.values) >= self.warmup_windows:
                self.mean = float(np.mean(self.values))
                self.std = float(np.std(self.values))

                # Schutz gegen std = 0
                if self.std <= 0:
                    self.std = 1.0

                self.ready = True

            return None

        # -------------------------------------------------
        # 2) Z-Score Normalisierung
        # -------------------------------------------------
        z = (x - self.mean) / (self.std + self.eps)

        # -------------------------------------------------
        # 3) Optional: Clamp
        # -------------------------------------------------
        if self.clip is not None:
            z = float(np.clip(z, -self.clip, self.clip))

        # -------------------------------------------------
        # 4) Optional: exponentielle Glättung
        # -------------------------------------------------
        if self.smooth_alpha is not None:
            if self._smooth_val is None:
                self._smooth_val = z
            else:
                a = self.smooth_alpha
                self._smooth_val = (1.0 - a) * self._smooth_val + a * z
            return self._smooth_val

        return z
