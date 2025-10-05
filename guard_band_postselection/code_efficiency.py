from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd

from .ldpc_simple_model import (
    SimpleLDPCParams,
    fit_ldpc_speed_params,
    eta_c_from_e_Smin,
    load_params,
    S_max_from_e,
    eta_c_fixed_speed,
)

__all__ = [
    "LDPCSimpleModel",
    "SimpleLDPCParams",
    "fit_ldpc_speed_params",
    "eta_c_from_e_Smin",
    "eta_c_fixed_speed",
    "S_max_from_e",
    "make_eta_c_provider",
]


class LDPCSimpleModel:
    """Convenience wrapper around the simple analytic LDPC throughput model."""

    def __init__(self, params: SimpleLDPCParams | None = None, *, fixed_speed: float | None = None) -> None:
        self._params: SimpleLDPCParams | None = params
        self._fixed_speed: float | None = None
        if fixed_speed is not None:
            self.set_fixed_speed(fixed_speed)

    @property
    def params(self) -> SimpleLDPCParams:
        if self._params is None:
            raise RuntimeError("LDPCSimpleModel parameters have not been initialised.")
        return self._params

    def is_ready(self) -> bool:
        return self._params is not None

    def fit_from_dataframe(self, df: pd.DataFrame, **fit_kwargs) -> SimpleLDPCParams:
        params = fit_ldpc_speed_params(df, **fit_kwargs)
        self._params = params
        return params

    def import_params(self, payload: SimpleLDPCParams | dict | str | Path) -> SimpleLDPCParams:
        if isinstance(payload, SimpleLDPCParams):
            params = payload
        elif isinstance(payload, (str, Path)):
            params = load_params(payload)
        elif isinstance(payload, dict):
            params = SimpleLDPCParams(**payload)
        else:
            raise TypeError("Unsupported payload type for import_params")
        self._params = params
        return params

    # --- Speed configuration -------------------------------------------------
    def set_fixed_speed(self, speed: float | None) -> None:
        """Optionally cache a constant decoder speed requirement (Mb/s)."""
        if speed is None:
            self._fixed_speed = None
        else:
            if speed < 0:
                raise ValueError("Decoder speed must be non-negative")
            self._fixed_speed = float(speed)

    def get_fixed_speed(self) -> float | None:
        return self._fixed_speed

    # --- Evaluation helpers --------------------------------------------------
    def eta_c(self, e_percent: Iterable[float], Smin: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        return eta_c_from_e_Smin(e_percent, Smin, self.params)

    def eta_c_fixed_speed(self, e_percent: Iterable[float], speed: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate eta_c using a constant decoder speed constraint."""
        if speed is None:
            if self._fixed_speed is None:
                raise RuntimeError("No fixed speed provided; call set_fixed_speed or pass speed explicitly")
            speed = self._fixed_speed
        return eta_c_fixed_speed(e_percent, speed, self.params)

    def s_max(self, e_percent: Iterable[float]) -> np.ndarray:
        return S_max_from_e(e_percent, self.params)

    def to_dict(self) -> dict:
        return self.params.to_dict()

    def to_json(self, path: Path | str) -> None:
        self.params.to_json(Path(path))


# Backwards compatible module-level helpers for quick access
fit_ldpc_speed_params_df = fit_ldpc_speed_params
eta_c_from_params = eta_c_from_e_Smin
eta_c_fixed_speed_from_params = eta_c_fixed_speed
S_max_from_params = S_max_from_e


def make_eta_c_provider(*, constant: float | None = None, ldpc_model: "LDPCSimpleModel" | None = None, speed: float | None = None) -> Callable[[Iterable[float]], np.ndarray]:
    """Return a callable that maps error rates (fractions) to coding efficiencies."""
    if (constant is None) == (ldpc_model is None):
        raise ValueError("provide exactly one of constant or ldpc_model")
    if constant is not None:
        value = float(constant)
        if not (0.0 < value <= 1.0):
            raise ValueError("constant eta_c must lie in (0, 1]")
        def provider(errors: Iterable[float]) -> np.ndarray:
            arr = np.asarray(errors, dtype=float)
            return np.full(arr.shape, value, dtype=float)
        return provider
    if ldpc_model is None:
        raise ValueError("ldpc_model must be provided when constant is None")
    def provider(errors: Iterable[float]) -> np.ndarray:
        arr = np.asarray(errors, dtype=float)
        eta_vals, _ = ldpc_model.eta_c_fixed_speed(arr * 100.0, speed)
        return np.asarray(eta_vals, dtype=float)
    return provider
