from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler


@dataclass
class GasContextResidualizer:
    ridge: Ridge
    scaler: RobustScaler
    feature_columns: list[str]

    @staticmethod
    def design_matrix(frame: pd.DataFrame) -> np.ndarray:
        gas_codes = frame["gas_code"].astype(int).to_numpy()
        concentration = np.log1p(frame["concentration"].astype(float).to_numpy())[:, None]
        one_hot = np.zeros((len(frame), 6), dtype=np.float64)
        valid = (gas_codes >= 1) & (gas_codes <= 6)
        one_hot[np.arange(len(frame))[valid], gas_codes[valid] - 1] = 1.0
        interactions = one_hot * concentration
        return np.concatenate([one_hot, concentration, interactions], axis=1)

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: list[str], ridge_alpha: float = 1.0) -> "GasContextResidualizer":
        design = cls.design_matrix(frame)
        target = frame[feature_columns].to_numpy(dtype=np.float64)
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(design, target)
        residual = target - ridge.predict(design)
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(residual)
        return cls(ridge=ridge, scaler=scaler, feature_columns=feature_columns)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        design = self.design_matrix(frame)
        target = frame[self.feature_columns].to_numpy(dtype=np.float64)
        residual = target - self.ridge.predict(design)
        transformed = self.scaler.transform(residual)
        return transformed.astype(np.float32)


@dataclass
class RobustSeriesScaler:
    scaler: RobustScaler

    @classmethod
    def fit(cls, array: np.ndarray) -> "RobustSeriesScaler":
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(array)
        return cls(scaler=scaler)

    def transform(self, array: np.ndarray) -> np.ndarray:
        return self.scaler.transform(array).astype(np.float32)


def feature_columns(frame: pd.DataFrame, prefix: str = "f") -> list[str]:
    return sorted([column for column in frame.columns if column.startswith(prefix)])
