"""Multidimensional reconciliation helper built on the key efficiency base."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .key_efficiency_base import KeyEfficiencyBase


class MD(KeyEfficiencyBase):
    """Virtual BIAWGNC based reconciliation scheme."""

    def __init__(
        self,
        modulation_variance: float,
        transmittance: float,
        excess_noise: float,
        *,
        code_efficiency: float | Callable[[np.ndarray], np.ndarray] | None = 0.95,
        Delta_QCT: float = 0.0,
        progress_bar: bool = False,
        shot_noise: float = 1.0,
    ) -> None:
        super().__init__(
            modulation_variance,
            transmittance,
            excess_noise,
            Delta_QCT=Delta_QCT,
            leak_strategy="error-rate",
            progress_bar=progress_bar,
            shot_noise=shot_noise,
        )
        self.set_code_efficiency(code_efficiency if code_efficiency is not None else 0.95)
        self.gaussian_attack_holevo_information = self._evaluate_holevo_information()

    def evaluate_key_rate_in_bits_per_pulse(self, *, code_efficiency=None) -> float:
        metrics = self.evaluate_reconciliation_efficiency(code_efficiency=code_efficiency)
        return metrics["key_rate"]

    def evaluate_reconciliation_efficiency(self, *, code_efficiency=None) -> dict[str, float]:
        """Compute eta using the error-rate leakage accounting from the notes.

        MD targets one bit per raw symbol: B_tgt = 1.
        With a virtual BSC of error e, the practical code rate is
        R = eta_c(e) * [1 - h2(e)]. The leakage is L = 1 - R, so the
        recovered private bits are B_priv = B_tgt - L = R.
        """
        snr = max(self.SNR, 0.0)
        raw_error = 0.5 * math.erfc(math.sqrt(snr))
        raw_error = float(np.clip(raw_error, 0.0, 1.0))
        effective_error = self._effective_bsc_error(raw_error)
        capacity = self._bsc_capacity(effective_error)
        coding_efficiency = float(
            self._evaluate_code_efficiency(np.array([effective_error]), override=code_efficiency)[0]
        )
        code_rate = float(coding_efficiency * capacity)  # R
        bits_sent = 1.0  # H(T(X)) for a balanced MD bit
        leak_per_bit = max(1.0 - code_rate, 0.0)
        bits_leaked = leak_per_bit
        bits_priv = bits_sent - bits_leaked  # = R
        denominator = max(self.I_AB, 1e-12)
        eta = bits_priv / denominator if denominator > 0.0 else 0.0
        key_rate = bits_priv - self._holevo_with_qct()
        return {
            "eta": float(eta),
            "bits_sent": float(bits_sent),
            "bits_leaked": float(bits_leaked),
            "raw_error_rate": float(raw_error),
            "error_rate": float(effective_error),
            "capacity": float(capacity),
            "coding_efficiency": float(coding_efficiency),
            "leak_per_bit": float(leak_per_bit),
            "code_rate": float(code_rate),
            "key_rate": float(key_rate),
            "I_AB": float(self.I_AB),
        }

    def _leak_from_error_rate(
        self, *, error_rate: float, coding_efficiency: float | None = None, **_kwargs
    ) -> float:
        effective_error = self._effective_bsc_error(error_rate)
        capacity = self._bsc_capacity(effective_error)
        if coding_efficiency is None:
            coding_efficiency = float(
                self._evaluate_code_efficiency(np.array([effective_error]))[0]
            )
        else:
            coding_efficiency = float(np.asarray(coding_efficiency, dtype=float))
        leak = np.clip(1.0 - coding_efficiency * capacity, 0.0, 1.0)
        return float(leak)

    def evaluate_slepian_wolf_leakage(self, *_, **__):
        """MD does not use a Slepian-Wolf term; placeholder for symmetry."""
        raise NotImplementedError(
            "MD scheme uses the BIAWGNC reconciliation model; no Slepian-Wolf leakage implemented yet."
        )

    def _biawgn_capacity(self, snr: float) -> float:
        """Numeric capacity of the virtual binary-input AWGN channel."""
        if snr <= 0.0:
            return 0.0
        nodes, weights = np.polynomial.hermite.hermgauss(80)
        scaled = np.sqrt(2.0) * nodes
        exponent = -2.0 * np.sqrt(snr) * scaled - 2.0 * snr
        exponent = np.clip(exponent, -120.0, 120.0)
        log_term = np.log1p(np.exp(exponent)) / np.log(2.0)
        integral = (1.0 / np.sqrt(np.pi)) * np.dot(weights, log_term)
        capacity = 1.0 - float(integral)
        if capacity < 0.0:
            return 0.0
        if capacity > 1.0:
            return 1.0
        return capacity
