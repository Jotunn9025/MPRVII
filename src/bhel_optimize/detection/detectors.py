"""Detector zoo: TensorFlow models plus cheap statistical/sklearn baselines.

Every detector implements the same tiny interface::

    detector.fit(windows)      # windows: (n, lookback, n_tags) float32
    scores = detector.score(windows)   # (n,) higher == more anomalous

Windowed detectors are trained on **normal data only** (unsupervised anomaly
detection), so no fault labels are needed for training — the simulated faults
are used only for weighting, threshold calibration and evaluation.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from bhel_optimize import config as C

#: MAD-based detectors must never divide by a window-local spread that happens
#: to be ~0 on a flat stretch (that produced 1e7 scores on normal data). The
#: denominator is floored at a fraction of the per-tag reference spread learned
#: from training data.
SCALE_FLOOR_FRACTION = 0.25

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

DEFAULT_EPOCHS = 25
DEFAULT_BATCH = 256


class BaseDetector(ABC):
    """Common interface for all anomaly detectors."""

    key: str = "base"
    name: str = "Base"
    kind: str = "statistical"  # tensorflow | sklearn | statistical
    windowed: bool = True

    def __init__(self, lookback: int, n_tags: int, seed: int = 0) -> None:
        self.lookback = lookback
        self.n_tags = n_tags
        self.seed = seed
        self.fitted = False

    # -- helpers ---------------------------------------------------------
    def _flat(self, windows: np.ndarray) -> np.ndarray:
        return windows.reshape(len(windows), -1)

    # -- interface -------------------------------------------------------
    def fit(self, windows: np.ndarray) -> "BaseDetector":
        self.fitted = True
        return self

    @abstractmethod
    def score(self, windows: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def save(self, directory) -> None:
        """Persist trained state (no-op for stateless detectors)."""

    def load(self, directory) -> None:
        """Restore trained state (no-op for stateless detectors)."""


# --------------------------------------------------------------------------
# Cheap baselines (no training, causal)
# --------------------------------------------------------------------------


class _MADWindowDetector(BaseDetector):
    """Shared plumbing for window-local robust z-score detectors.

    ``fit`` learns a per-tag *reference scale* (median window MAD on normal
    training windows) so a flat stretch cannot blow the score up.
    """

    tail: int = 15

    def __init__(self, lookback: int, n_tags: int, seed: int = 0, tail: int | None = None):
        super().__init__(lookback, n_tags, seed)
        if tail is not None:
            self.tail = tail
        self.reference_scale: np.ndarray | None = None

    # -- helpers ---------------------------------------------------------
    def _history(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split windows into (recent tail, preceding history)."""
        return windows[:, -self.tail :, :], windows[:, : -self.tail, :] if self.tail < windows.shape[1] else windows[:, :-1, :]

    def _window_scale(self, windows: np.ndarray) -> np.ndarray:
        """Per-window, per-tag robust spread of the preceding history (n, tags)."""
        _, history = self._history(windows)
        median = np.median(history, axis=1)
        mad = np.median(np.abs(history - median[:, None, :]), axis=1)
        return 1.4826 * mad

    def fit(self, windows: np.ndarray) -> "_MADWindowDetector":
        scales = self._window_scale(windows)
        self.reference_scale = np.median(scales, axis=0) + 1e-9
        self.fitted = True
        return self

    def _floored_scale(self, windows: np.ndarray) -> np.ndarray:
        scales = self._window_scale(windows)
        reference = (
            self.reference_scale
            if self.reference_scale is not None
            else np.median(scales, axis=0) + 1e-9
        )
        return np.maximum(scales, SCALE_FLOOR_FRACTION * reference)


class EWMAMADDetector(_MADWindowDetector):
    """Single-sample deviation of the last reading from the window baseline."""

    key = "ewma_mad"
    name = "EWMA / MAD (sample deviation)"
    kind = "statistical"
    tail = 1

    def score(self, windows: np.ndarray) -> np.ndarray:
        last = windows[:, -1, :]
        history = windows[:, :-1, :]
        median = np.median(history, axis=1)
        z = np.abs(last - median) / self._floored_scale(windows)
        return z.max(axis=1)


class MeanShiftDetector(_MADWindowDetector):
    """Control-chart style test for a *sustained* shift of the recent tail.

    Denominators are floored so flat stretches cannot inflate the statistic.
    This is the detector that catches creeping faults whose per-sample
    deviation is smaller than ordinary noise.
    """

    key = "mean_shift"
    name = "Windowed mean-shift (control chart)"
    kind = "statistical"
    tail = 15

    def score(self, windows: np.ndarray) -> np.ndarray:
        tail, history = self._history(windows)
        median = np.median(history, axis=1)
        scale = self._floored_scale(windows) / np.sqrt(self.tail)
        z = np.abs(tail.mean(axis=1) - median) / scale
        return z.max(axis=1)


class StalenessDetector(BaseDetector):
    """Detects frozen/stuck sensors: low variance over the recent window.

    Deviation-based detectors cannot see a stuck sensor (a frozen value sits
    inside the normal range), so the boiler failure-mode list needs this
    dedicated variance-collapse detector.
    """

    key = "staleness"
    name = "Staleness / variance collapse"
    kind = "statistical"

    def __init__(self, lookback: int, n_tags: int, seed: int = 0, tail: int = 15):
        super().__init__(lookback, n_tags, seed)
        self.tail = tail
        self.reference: float | None = None

    def fit(self, windows: np.ndarray) -> "StalenessDetector":
        spread = self._spread(windows)
        self.reference = float(np.median(spread)) + 1e-9
        self.fitted = True
        return self

    def _spread(self, windows: np.ndarray) -> np.ndarray:
        tail = windows[:, -self.tail :, :]
        # smallest per-tag spread across the window tail
        return tail.std(axis=1).max(axis=1)

    def score(self, windows: np.ndarray) -> np.ndarray:
        spread = self._spread(windows)
        reference = self.reference if self.reference else (float(np.median(spread)) + 1e-9)
        return np.log((reference + 1e-9) / (spread + 1e-9))


class IsolationForestDetector(BaseDetector):
    """Isolation Forest on flattened windows (sklearn baseline)."""

    key = "isolation_forest"
    name = "Isolation Forest"
    kind = "sklearn"

    def __init__(self, lookback: int, n_tags: int, seed: int = 0, n_estimators: int = 200):
        super().__init__(lookback, n_tags, seed)
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, windows: np.ndarray) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(256, len(windows)),
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(self._flat(windows))
        self.fitted = True
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("detector not fitted")
        return -self.model.score_samples(self._flat(windows))


class OCSVMDetector(BaseDetector):
    """One-Class SVM on standardised flattened windows (sklearn baseline)."""

    key = "ocsvm"
    name = "One-Class SVM"
    kind = "sklearn"

    def __init__(self, lookback: int, n_tags: int, seed: int = 0, max_fit: int = 3000):
        super().__init__(lookback, n_tags, seed)
        self.max_fit = max_fit
        self.model = None
        self.scaler = None

    def fit(self, windows: np.ndarray) -> "OCSVMDetector":
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        flat = self._flat(windows)
        if len(flat) > self.max_fit:
            idx = np.linspace(0, len(flat) - 1, self.max_fit).astype("int64")
            flat = flat[idx]
        self.scaler = StandardScaler().fit(flat)
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        self.model.fit(self.scaler.transform(flat))
        self.fitted = True
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("detector not fitted")
        return -self.model.decision_function(self.scaler.transform(self._flat(windows)))


# --------------------------------------------------------------------------
# TensorFlow autoencoder family
# --------------------------------------------------------------------------


class BaseAutoencoder(BaseDetector):
    """Shared Keras plumbing for reconstruction / prediction residual models."""

    kind = "tensorflow"
    early_stop_patience = 4

    def __init__(
        self,
        lookback: int,
        n_tags: int,
        seed: int = 0,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH,
    ):
        super().__init__(lookback, n_tags, seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history: dict = {}

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, windows: np.ndarray) -> "BaseAutoencoder":
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        self.model = self._build()
        x, y = self._supervised_pair(windows)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=self.early_stop_patience, restore_best_weights=True
            )
        ]
        history = self.model.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self.fitted = True
        return self

    def _supervised_pair(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Default: reconstruct the whole window."""
        return windows, windows

    def _residual(self, windows: np.ndarray) -> np.ndarray:
        x, y = self._supervised_pair(windows)
        pred = self.model.predict(x, batch_size=1024, verbose=0)
        return ((pred - y) ** 2).reshape(len(windows), -1)

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("detector not fitted")
        return self._residual(windows).mean(axis=1)

    def save(self, directory) -> None:
        from pathlib import Path

        if self.model is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.model.save(directory / f"{self.key}.keras")

    def load(self, directory) -> None:
        from pathlib import Path

        import tensorflow as tf

        path = Path(directory) / f"{self.key}.keras"
        self.model = tf.keras.models.load_model(path)
        self.fitted = True


class DenseAutoencoder(BaseAutoencoder):
    """Flattened-window dense autoencoder (correlation-based detection)."""

    key = "dense_ae"
    name = "Dense autoencoder"

    def _supervised_pair(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = self._flat(windows)
        return flat, flat

    def _build(self):
        from tensorflow import keras

        flat = self.lookback * self.n_tags
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(flat,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(flat),
            ],
            name=self.key,
        )
        model.compile(optimizer="adam", loss="mse")
        return model


class TCNAutoencoder(BaseAutoencoder):
    """1D temporal-convolution autoencoder (Opt-TCAE-style, tube-leak candidate)."""

    key = "tcn_ae"
    name = "TCN autoencoder"

    def _build(self):
        from tensorflow import keras

        inp = keras.layers.Input(shape=(self.lookback, self.n_tags))
        x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
        x = keras.layers.Conv1D(16, 3, padding="same", activation="relu")(x)
        x = keras.layers.Conv1D(16, 3, padding="same", activation="relu")(x)
        x = keras.layers.Conv1D(self.n_tags, 3, padding="same")(x)
        model = keras.Model(inp, x, name=self.key)
        model.compile(optimizer="adam", loss="mse")
        return model


class LSTMPredictor(BaseAutoencoder):
    """LSTM next-step predictor; residual on the last sample of each window."""

    key = "lstm_residual"
    name = "LSTM prediction residual"

    def _supervised_pair(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return windows[:, :-1, :], windows[:, -1, :]

    def _residual(self, windows: np.ndarray) -> np.ndarray:
        x, y = self._supervised_pair(windows)
        pred = self.model.predict(x, batch_size=1024, verbose=0)
        return (pred - y) ** 2

    def _build(self):
        from tensorflow import keras

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.lookback - 1, self.n_tags)),
                keras.layers.LSTM(32, activation="tanh"),
                keras.layers.Dense(self.n_tags),
            ],
            name=self.key,
        )
        model.compile(optimizer="adam", loss="mse")
        return model


class TransformerForecaster(BaseAutoencoder):
    """Small self-attention forecaster; heavy, enable only where it wins."""

    key = "transformer_residual"
    name = "Transformer forecaster residual"
    heavy = True
    early_stop_patience = 3

    def _supervised_pair(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return windows[:, :-1, :], windows[:, -1, :]

    def _residual(self, windows: np.ndarray) -> np.ndarray:
        x, y = self._supervised_pair(windows)
        pred = self.model.predict(x, batch_size=1024, verbose=0)
        return (pred - y) ** 2

    def _build(self):
        from tensorflow import keras

        seq = self.lookback - 1
        inp = keras.layers.Input(shape=(seq, self.n_tags))
        proj = keras.layers.Dense(32)(inp)
        attn = keras.layers.MultiHeadAttention(num_heads=2, key_dim=8)(proj, proj)
        x = keras.layers.LayerNormalization()(attn + proj)
        x = keras.layers.GlobalAveragePooling1D()(x)
        out = keras.layers.Dense(self.n_tags)(x)
        model = keras.Model(inp, out, name=self.key)
        model.compile(optimizer="adam", loss="mse")
        return model


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

DEFAULT_DETECTOR_KEYS: tuple[str, ...] = (
    "ewma_mad",
    "mean_shift",
    "staleness",
    "dense_ae",
    "lstm_residual",
    "tcn_ae",
    "isolation_forest",
    "ocsvm",
)

DETECTOR_CLASSES: dict[str, type[BaseDetector]] = {
    cls.key: cls
    for cls in (
        EWMAMADDetector,
        MeanShiftDetector,
        StalenessDetector,
        DenseAutoencoder,
        LSTMPredictor,
        TCNAutoencoder,
        TransformerForecaster,
        IsolationForestDetector,
        OCSVMDetector,
    )
}


def detector_keys(include_heavy: bool = False) -> tuple[str, ...]:
    """Default candidate set, optionally including heavy models."""
    keys = list(DEFAULT_DETECTOR_KEYS)
    if include_heavy:
        keys.append("transformer_residual")
    return tuple(keys)


def build_detector(key: str, lookback: int, n_tags: int, seed: int = 0, **kwargs) -> BaseDetector:
    """Instantiate a detector by registry key."""
    if key not in DETECTOR_CLASSES:
        raise KeyError(f"unknown detector: {key} (have {sorted(DETECTOR_CLASSES)})")
    return DETECTOR_CLASSES[key](lookback=lookback, n_tags=n_tags, seed=seed, **kwargs)
