from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize

__all__ = [
    "SimpleLDPCParams",
    "fit_ldpc_speed_params",
    "eta_c_from_e_Smin",
    "eta_c_fixed_speed",
    "S_max_from_e",
    "speed_model_from_params",
    "load_params",
]

GAMMA_MIN = 0.5


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def capacity_from_ber_percent(ber_percent: Iterable[float]) -> np.ndarray:
    e = np.asarray(ber_percent, dtype=float) / 100.0
    e = np.clip(e, 1e-12, 0.5 - 1e-12)
    return 1.0 - binary_entropy(e)


@dataclass
class SimpleLDPCParams:
    """Container for the simple analytic LDPC throughput parameters."""

    eta_inf: float
    s0: float
    gamma: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _prepare_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    rate_cols = [c for c in df.columns if c.lower().startswith("code rate")]
    if not rate_cols:
        raise ValueError("DataFrame must contain columns starting with 'code rate'.")

    def parse_rate(label: str) -> float:
        frac = label.split("=")[1].strip()
        return float(Fraction(frac))

    tidy = (
        df.melt(id_vars="BER", value_vars=rate_cols, var_name="rate_label", value_name="speed")
        .assign(
            rate=lambda d: d["rate_label"].map(parse_rate),
            ber_fraction=lambda d: (d["BER"] / 100.0).clip(1e-12, 0.5 - 1e-12),
            capacity=lambda d: 1.0 - binary_entropy(d["ber_fraction"]),
        )
        .sort_values(["rate", "BER"])
        .reset_index(drop=True)
    )
    return tidy


def speed_model_from_params(R: np.ndarray, capacity: np.ndarray, params: SimpleLDPCParams) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    gap = np.maximum(capacity - R / params.eta_inf, 0.0)
    return params.s0 * np.power(gap, params.gamma)


def _objective_simple(
    theta: np.ndarray,
    R_arr: np.ndarray,
    C_arr: np.ndarray,
    S_arr: np.ndarray,
    ok_mask: np.ndarray,
    stall_mask: np.ndarray,
    lam: float,
    eps: float,
    rho: float,
) -> float:
    eta_inf = 1.0 / (1.0 + np.exp(-theta[0]))
    s0 = np.exp(theta[1])
    gamma = np.exp(theta[2]) + GAMMA_MIN

    params = SimpleLDPCParams(eta_inf=eta_inf, s0=s0, gamma=gamma)

    pred = speed_model_from_params(R_arr, C_arr, params)
    mse = np.mean((pred[ok_mask] - S_arr[ok_mask]) ** 2)

    penalty = 0.0
    if np.any(stall_mask):
        c_th = R_arr[stall_mask] / eta_inf
        margin = C_arr[stall_mask] - (c_th - eps)
        penalty = lam * np.mean(np.maximum(margin, 0.0) ** 2)

    reg = rho * (gamma - 1.0) ** 2
    return mse + penalty + reg


def fit_ldpc_speed_params(
    df: pd.DataFrame,
    *,
    lam: float = 50.0,
    eps: float = 2e-3,
    rho: float = 1e-4,
) -> SimpleLDPCParams:
    """Fit the simple analytic LDPC throughput model parameters from a raw measurement table."""

    tidy = _prepare_long_frame(df)
    R_arr = tidy["rate"].to_numpy(dtype=float)
    C_arr = tidy["capacity"].to_numpy(dtype=float)
    S_arr = tidy["speed"].to_numpy(dtype=float)

    ok_mask = np.isfinite(S_arr) & (S_arr > 0)
    stall_mask = ~ok_mask

    s0_guess = float(np.nanmedian(S_arr[ok_mask])) if np.any(ok_mask) else 1.0

    theta0 = np.array([
        np.log(0.9 / (1 - 0.9)),
        np.log(s0_guess),
        np.log(0.5),
    ], dtype=float)

    result = optimize.minimize(
        _objective_simple,
        theta0,
        method="L-BFGS-B",
        args=(R_arr, C_arr, S_arr, ok_mask, stall_mask, lam, eps, rho),
        options={"maxiter": 2000, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Parameter fit did not converge: {result.message}")

    eta_inf = 1.0 / (1.0 + np.exp(-result.x[0]))
    s0 = float(np.exp(result.x[1]))
    gamma = float(np.exp(result.x[2]) + GAMMA_MIN)
    return SimpleLDPCParams(eta_inf=eta_inf, s0=s0, gamma=gamma)


def eta_c_from_e_Smin(
    e_percent: Iterable[float],
    Smin: Iterable[float],
    params: SimpleLDPCParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the simple-model coding efficiency and optimal rate."""

    e_percent = np.asarray(e_percent, dtype=float)
    Smin = np.asarray(Smin, dtype=float)
    ce = capacity_from_ber_percent(e_percent)
    ce_pow = np.power(ce, params.gamma)
    denom = params.s0 * ce_pow

    ratio = np.maximum(Smin / denom, 0.0)
    term = np.power(ratio, 1.0 / params.gamma)
    eta = params.eta_inf * np.maximum(1.0 - term, 0.0)
    eta[ce <= 0] = 0.0
    R_star = eta * ce
    return eta, R_star


def eta_c_fixed_speed(
    e_percent: Iterable[float],
    speed: Union[float, Iterable[float]],
    params: SimpleLDPCParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised closed-form eta evaluation for constant decoder speed."""

    e_percent = np.asarray(e_percent, dtype=float)
    speed = np.asarray(speed, dtype=float)
    ce = capacity_from_ber_percent(e_percent)
    ce_pow = np.power(ce, params.gamma)
    denom = params.s0 * ce_pow
    ratio = np.maximum(speed / denom, 0.0)
    term = np.power(ratio, 1.0 / params.gamma)
    eta = params.eta_inf * np.maximum(1.0 - term, 0.0)
    eta[ce <= 0] = 0.0
    R_star = eta * ce
    return eta, R_star


def S_max_from_e(e_percent: Iterable[float], params: SimpleLDPCParams) -> np.ndarray:
    ce = capacity_from_ber_percent(e_percent)
    ce = np.clip(ce, 0.0, None)
    return params.s0 * np.power(ce, params.gamma)


def load_params(path: Path | str) -> SimpleLDPCParams:
    payload = json.loads(Path(path).read_text())
    field_names = {field.name for field in fields(SimpleLDPCParams)}
    filtered = {key: payload[key] for key in field_names if key in payload}
    missing = [name for name in field_names if name not in filtered]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing parameters in {path}: {missing_str}")
    return SimpleLDPCParams(**filtered)
