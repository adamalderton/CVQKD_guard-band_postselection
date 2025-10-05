"""Shared base machinery for CVQKD key-efficiency computations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable, List, Sequence, Literal

import numpy as np
from scipy.stats import multivariate_normal, norm, mvn

__all__ = ["KeyEfficiencyBase"]

LeakStrategy = Literal["slepian-wolf", "error-rate", "capacity"]
_LEAK_STRATEGIES = {"slepian-wolf", "error-rate", "capacity"}


@lru_cache(maxsize=None)
def _rect_prob(mean_x: float, mean_y: float,
               c00: float, c01: float, c10: float, c11: float,
               x1: float, x2: float, y1: float, y2: float) -> float:
    """Probability that (X, Y) lies in the rectangle (x1, x2] x (y1, y2]."""
    mean = np.array([mean_x, mean_y], dtype=float)
    cov = np.array([[c00, c01],
                    [c10, c11]], dtype=float)

    lower = np.array([x1, y1], dtype=float)
    upper = np.array([x2, y2], dtype=float)

    p, info = mvn.mvnun(lower, upper, mean, cov)
    if info != 0:
        raise RuntimeError(f"mvnun did not converge (info={info})")
    return float(p)


class KeyEfficiencyBase(ABC):
    """Channel bookkeeping shared by the different key-efficiency schemes."""

    def __init__(
        self,
        modulation_variance: float,
        transmittance: float,
        excess_noise: float,
        Delta_QCT: float = 0.0,
        *,
        leak_strategy: str = "error-rate",
        progress_bar: bool = False,
        shot_noise: float = 1.0,
    ) -> None:
        self.progress_bar = progress_bar

        self.modulation_variance = float(modulation_variance)
        self.transmittance = float(transmittance)
        self.excess_noise = float(excess_noise)
        self.shot_noise = float(shot_noise)

        self.Delta_QCT = float(Delta_QCT)
        self.leak_strategy = self._validate_leak_strategy(leak_strategy)

        self.bob_variance = (
            0.5 * self.transmittance * self.modulation_variance
            + self.shot_noise
            + 0.5 * self.excess_noise
        )

        # In the case of dual-homodyne (heterodyne) detection, see Laudenbach-2018.
        self.SNR = (
            0.5 * self.transmittance * self.modulation_variance
            / (self.shot_noise + 0.5 * self.excess_noise)
        )
        
        # This is in the case of NO POST-SELECTION.
        self.I_AB = np.log2(1.0 + self.SNR)

        # These correspond to a^EB, b^E_B, c^E_B in Laudenbach-2018.
        self.a = self.modulation_variance + 1.0
        self.b = (2.0 * self.bob_variance) - self.shot_noise
        self.c = np.sqrt(self.transmittance * (self.modulation_variance ** 2 + 2.0 * self.modulation_variance))

        self.cov_mat_EB = np.array(
            [
                [self.a, self.c],
                [self.c, self.b],
            ],
            dtype=float,
        )

        self.cov_mat = self._build_joint_covariance()

        self.px_rv = norm(loc=0.0, scale=np.sqrt(self.modulation_variance))
        self.py_rv = norm(loc=0.0, scale=np.sqrt(self.bob_variance))
        self.joint_rv = multivariate_normal(mean=np.zeros(2), cov=self.cov_mat)

        Q_star_cov_mat = self.cov_mat + self.shot_noise * np.eye(2)
        self.Q_star_rv = multivariate_normal(mean=np.zeros(2), cov=Q_star_cov_mat)

        self._code_efficiency_provider = self._normalise_code_efficiency(0.95)

        self.p_pass = None
        self.a_PS = None
        self.b_PS = None
        self.c_PS = None
        self.cov_mat_PS = None

        self.px_Q_PS_values = None
        self.py_Q_PS_values = None
        self.Q_PS_values = None

    def _normalise_code_efficiency(self, provider) -> Callable[[np.ndarray], np.ndarray]:
        if isinstance(provider, (int, float)):
            value = float(provider)
            if not (0.0 < value <= 1.0):
                raise ValueError("code efficiency must lie in (0, 1]")
            def constant(errors: np.ndarray) -> np.ndarray:
                errors = np.asarray(errors, dtype=float)
                return np.full(errors.shape, value, dtype=float)
            return constant
        if callable(provider):
            def wrapped(errors: np.ndarray) -> np.ndarray:
                arr = np.asarray(errors, dtype=float)
                result = provider(arr)
                result = np.asarray(result, dtype=float)
                if result.shape != arr.shape:
                    raise ValueError("code efficiency provider must preserve shape")
                return result
            return wrapped
        raise TypeError("code efficiency provider must be a float or callable")

    def set_code_efficiency(self, provider) -> None:
        self._code_efficiency_provider = self._normalise_code_efficiency(provider)

    def _evaluate_code_efficiency(self, errors: np.ndarray, override=None) -> np.ndarray:
        resolver = self._code_efficiency_provider if override is None else self._normalise_code_efficiency(override)
        eta = resolver(errors)
        return np.clip(eta, 0.0, 1.0)

    def _validate_leak_strategy(self, strategy: str) -> str:
        strategy_normalised = strategy.lower()
        if strategy_normalised not in _LEAK_STRATEGIES:
            allowed = ", ".join(sorted(_LEAK_STRATEGIES))
            raise ValueError(f"leak_strategy must be one of {allowed}; received {strategy!r}")
        return strategy_normalised

    def set_leak_strategy(self, strategy: str) -> None:
        """Update the reconciliation leakage strategy at runtime."""
        self.leak_strategy = self._validate_leak_strategy(strategy)

    def _compute_reconciliation_leak(self, **kwargs) -> float:
        """Dispatch to the configured reconciliation leakage model."""
        if self.leak_strategy == "slepian-wolf":
            return self._leak_from_slepian_wolf(**kwargs)
        if self.leak_strategy == "error-rate":
            return self._leak_from_error_rate(**kwargs)
        if self.leak_strategy == "capacity":
            return self._leak_from_capacity(**kwargs)
        raise ValueError(f"Unsupported leak strategy: {self.leak_strategy}")

    def _leak_from_slepian_wolf(self, **kwargs) -> float:
        """Compute the reconciliation leak via Slepian-Wolf bounds."""
        raise NotImplementedError("TODO: implement the Slepian-Wolf leakage estimate.")

    def _leak_from_error_rate(self, **kwargs) -> float:
        """Compute the reconciliation leak from error rates."""
        raise NotImplementedError("TODO: implement the error-rate leakage model for this protocol.")

    def _leak_from_capacity(self, **kwargs) -> float:
        """Compute the reconciliation leak via channel capacity."""
        raise NotImplementedError("TODO: implement the capacity-based leakage model for this protocol.")

    def _build_joint_covariance(self) -> np.ndarray:
        """Covariance matrix for the prepare-and-measure Gaussian variables, in the case of dual-homodyne detection."""
        cross_cov = np.sqrt(self.transmittance / 2.0) * self.modulation_variance
        return np.array(
            [
                [self.modulation_variance, cross_cov],
                [cross_cov, self.bob_variance],
            ],
            dtype=float,
        )

    @abstractmethod
    def evaluate_key_rate_in_bits_per_pulse(self, *args, **kwargs) -> float:
        """Protocol-specific key rate."""
        raise NotImplementedError

    def evaluate_slepian_wolf_leakage(self, *args, **kwargs) -> float:
        """Placeholder for the Slepian-Wolf leakage term."""
        raise NotImplementedError(
            "TODO: implement the Slepian-Wolf leakage estimate for this protocol."
        )

    def evaluate_quantisation_entropy(self, normalised_tau_arr: Sequence[float]) -> float:
        """Entropy of the quantiser defined by the interval edges."""
        tau_arr = [
            normalised_tau_arr[i] * np.sqrt(self.modulation_variance)
            for i in range(len(normalised_tau_arr))
        ]

        num_intervals = getattr(self, "number_of_intervals", len(normalised_tau_arr) - 1)
        interval_probabilities = [
            self._integrate_1D_gaussian_pdf(self.px_rv, [tau_arr[i], tau_arr[i + 1]])
            for i in range(num_intervals)
        ]

        interval_probabilities = [p for p in interval_probabilities if p != 0.0]
        return -sum(p * np.log2(p) for p in interval_probabilities)

    def _evaluate_holevo_information(
        self,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
    ) -> float:
        """Evaluate the Holevo information using the symplectic eigenvalue method, for DUAL-HOMODYNE detection."""
        a_val = self.a if a is None else a
        b_val = self.b if b is None else b
        c_val = self.c if c is None else c

        sqrt_value = (a_val + b_val) ** 2 - (4.0 * c_val ** 2)

        nu_0 = 0.5 * (np.sqrt(sqrt_value) + (b_val - a_val))
        nu_1 = 0.5 * (np.sqrt(sqrt_value) - (b_val - a_val))
        nu_2 = a_val - (c_val ** 2) / (b_val + 1.0) # Extra 1 for dual-homodyne (0 for homodyne)

        return self._g(nu_0) + self._g(nu_1) - self._g(nu_2)

    def evaluate_devetak_winter(
        self,
        beta: float,
        *,
        mutual_information: float | None = None,
        holevo: float | None = None,
    ) -> float:
        """Compute the Devetak-Winter bound K_DW = beta * I_AB - chi."""
        if beta < 0.0:
            raise ValueError('beta must be non-negative')
        mi = self.I_AB if mutual_information is None else mutual_information
        chi = self._holevo_with_qct() if holevo is None else holevo
        return beta * mi - chi

    def _holevo_with_qct(
        self,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
    ) -> float:
        """Holevo information plus the QCT offset."""
        return self._evaluate_holevo_information(a, b, c) + self.Delta_QCT

    def _g(self, x: float) -> float:
        if x <= 1.0:
            return 0.0

        return ((x + 1.0) / 2.0) * np.log2((x + 1.0) / 2.0) - ((x - 1.0) / 2.0) * np.log2((x - 1.0) / 2.0)

    def _binary_entropy(self, e: float) -> float:
        if not 0.0 <= e <= 1.0:
            raise ValueError("Error rate e must be between 0 and 1 inclusive.")

        epsilon = 1e-15
        e = min(max(e, epsilon), 1 - epsilon)

        one_minus_e = 1.0 - e
        return -(e * np.log2(e) + one_minus_e * np.log2(one_minus_e))

    def _effective_bsc_error(self, error_rate: float) -> float:
        """Fold binary error probabilities into [0, 0.5] for capacity calculations."""
        error = float(np.clip(error_rate, 0.0, 1.0))
        return error if error <= 0.5 else 1.0 - error

    def _bsc_capacity(self, error_rate: float) -> float:
        """Return the symmetric-channel capacity 1 - h2(e)."""
        effective_error = self._effective_bsc_error(error_rate)
        return max(0.0, 1.0 - self._binary_entropy(effective_error))

    def _integrate_1D_gaussian_pdf(self, rv, lims: Sequence[float]) -> float:
        return rv.cdf(lims[1]) - rv.cdf(lims[0])

    def _integrate_2D_gaussian_pdf(self, rv, xlims: Sequence[float], ylims: Sequence[float]) -> float:
        if (
            np.isneginf(xlims[0])
            and np.isposinf(xlims[1])
            and np.isneginf(ylims[0])
            and np.isposinf(ylims[1])
        ):
            return 1.0

        x1, x2 = xlims
        y1, y2 = ylims
        mean_x, mean_y = rv.mean
        c00, c01, c10, c11 = rv.cov.ravel()

        return _rect_prob(mean_x, mean_y, c00, c01, c10, c11, x1, x2, y1, y2)

    def _generate_gray_bit_assignment(self, m: int) -> List[str]:
        return [f"{(i ^ (i >> 1)):0{m}b}" for i in range(2 ** m)]

    def _generate_binary_bit_assignment(self, m: int) -> List[str]:
        return [f"{i:0{m}b}" for i in range(2 ** m)]

    def _hamming_distance(self, bit_string_1: str, bit_string_2: str) -> int:
        return sum(ch1 != ch2 for ch1, ch2 in zip(bit_string_1, bit_string_2))








