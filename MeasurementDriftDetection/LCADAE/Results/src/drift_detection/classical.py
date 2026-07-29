from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .windows import flatten_windows


class PCASPEModel:
    def __init__(self, n_components: int = 20, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model: PCA | None = None

    def fit(self, windows: np.ndarray) -> "PCASPEModel":
        x = flatten_windows(windows)
        components = min(self.n_components, x.shape[0] - 1, x.shape[1])
        components = max(1, components)
        self.model = PCA(n_components=components, random_state=self.random_state)
        self.model.fit(x)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        x = flatten_windows(windows)
        reconstruction = self.model.inverse_transform(self.model.transform(x))
        return np.mean((x - reconstruction) ** 2, axis=1)


class IsolationForestModel:
    def __init__(self, n_estimators: int = 300, random_state: int = 42, n_jobs: int = -1):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, windows: np.ndarray) -> "IsolationForestModel":
        self.model.fit(flatten_windows(windows))
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(flatten_windows(windows))
