"""Sliced reconciliation protocol built on the key efficiency base."""

from __future__ import annotations

import numpy as np
import math
from functools import lru_cache
from typing import Callable

from scipy.special import ndtr

from .key_efficiency_base import KeyEfficiencyBase


@lru_cache(maxsize=None)
def _cached_hermgauss(points: int):
    if points <= 0:
        raise ValueError('quadrature_points must be positive')
    return np.polynomial.hermite.hermgauss(points)


class SR(KeyEfficiencyBase):
    """Van Assche et al. sliced reconciliation without guard bands."""

    def __init__(
        self,
        m: int,
        modulation_variance: float,
        transmittance: float,
        excess_noise: float,
        *,
        code_efficiency: float | Callable[[np.ndarray], np.ndarray] | None = 0.95,
        Delta_QCT: float = 0.0,
        leak_strategy: str = "error-rate",
        progress_bar: bool = False,
        shot_noise: float = 1.0,
    ) -> None:
        super().__init__(
            modulation_variance,
            transmittance,
            excess_noise,
            Delta_QCT=Delta_QCT,
            leak_strategy=leak_strategy,
            progress_bar=progress_bar,
            shot_noise=shot_noise,
        )

        self.set_code_efficiency(code_efficiency if code_efficiency is not None else 0.95)
        self.m = m
        self.number_of_intervals = 2 ** m
        self.slice_error_rates = None

        self.gaussian_attack_holevo_information = self._evaluate_holevo_information()

    def evaluate_key_rate_in_bits_per_pulse(self, tau_arr, g_arr=None, *, code_efficiency=None, bit_assignment="Gray") -> float:
        metrics = self.evaluate_reconciliation_efficiency(
            tau_arr,
            g_arr,
            code_efficiency=code_efficiency,
            bit_assignment=bit_assignment,
        )
        return metrics["key_rate"]

    def evaluate_key_rate_components(self, tau_arr, g_arr=None, *, code_efficiency=None, bit_assignment="Gray") -> dict[str, float]:
        metrics = self.evaluate_reconciliation_efficiency(
            tau_arr,
            g_arr,
            code_efficiency=code_efficiency,
            bit_assignment=bit_assignment,
        )
        holevo_information = self._holevo_with_qct()
        return {
            "key_rate": metrics["key_rate"],
            "bits_sent": metrics["bits_sent"],
            "bits_leaked": metrics["bits_leaked"],
            "eta": metrics["eta"],
            "holevo": holevo_information,
        }

    def evaluate_reconciliation_efficiency(
        self,
        tau_arr,
        g_arr=None,
        *,
        code_efficiency=None,
        bit_assignment="Gray",
    ) -> dict[str, float]:
        """Return the reconciliation efficiency components for m slices."""
        quantisation_entropy = self.evaluate_quantisation_entropy(tau_arr)
        slice_error_rates = self._evaluate_slice_error_rates(
            tau_arr, g_arr, bit_assignment=bit_assignment
        )
        raw_errors = np.asarray(slice_error_rates, dtype=float)
        effective_errors = np.array([self._effective_bsc_error(e) for e in raw_errors])
        capacities = np.array([self._bsc_capacity(e) for e in effective_errors])
        coding_efficiencies = self._evaluate_code_efficiency(effective_errors, override=code_efficiency)
        code_rates = np.asarray(coding_efficiencies * capacities, dtype=float)
        leak_terms = np.clip(1.0 - code_rates, 0.0, 1.0)
        bits_sent = quantisation_entropy
        bits_leaked = float(leak_terms.sum())
        numerator = bits_sent - bits_leaked
        denominator = max(self.I_AB, 1e-12)
        eta = numerator / denominator if denominator > 0.0 else 0.0
        key_rate = numerator - self._holevo_with_qct()
        slice_metrics = []
        for raw, eff, cap, eta_c, rate, leak in zip(
            raw_errors,
            effective_errors,
            capacities,
            coding_efficiencies,
            code_rates,
            leak_terms,
        ):
            slice_metrics.append(
                {
                    "raw_error_rate": float(raw),
                    "error_rate": float(eff),
                    "capacity": float(cap),
                    "coding_efficiency": float(eta_c),
                    "code_rate": float(rate),
                    "leak": float(leak),
                }
            )
        self.slice_error_rates = raw_errors
        return {
            "eta": float(eta),
            "bits_sent": float(bits_sent),
            "bits_leaked": float(bits_leaked),
            "key_rate": float(key_rate),
            "quantisation_entropy": float(quantisation_entropy),
            "slices": slice_metrics,
            "I_AB": self.I_AB,
        }

    def _evaluate_slice_error_rates(self, tau_arr, g_arr=None, bit_assignment="Gray"):
        """TODO: Change this to multi-round. It might already be????"""
        if g_arr is None:
            g_arr = np.zeros((len(tau_arr), 2))

        x_tau_arr = np.sqrt(self.modulation_variance) * np.array(tau_arr)
        y_tau_arr = np.sqrt(self.bob_variance) * np.array(tau_arr)
        y_g_arr = np.sqrt(self.bob_variance) * np.array(g_arr)

        if bit_assignment == "Gray":
            bit_strings = self._generate_gray_bit_assignment(self.m)
        elif bit_assignment == "binary":
            bit_strings = self._generate_binary_bit_assignment(self.m)
        else:
            raise ValueError("bit_assignment must be 'Gray' or 'binary'")

        # Decode slices from least-significant to most-significant so that each
        # stage benefits from the previously corrected slices, matching
        # Van Assche-2004 multi-stage decoding.
        bit_indices = list(range(self.m - 1, -1, -1))  # LSB -> MSB

        # Pre-compute interval probabilities for weighting prefixes.
        var_x = self.modulation_variance
        var_y = self.bob_variance
        cov_xy = np.sqrt(self.transmittance / 2.0) * self.modulation_variance

        sigma_y = np.sqrt(var_y)
        sigma_cond_sq = max(var_x - (cov_xy ** 2) / var_y, 1e-12)
        sigma_cond = np.sqrt(sigma_cond_sq)
        mu_factor = cov_xy / var_y

        nodes, weights = _cached_hermgauss(240)
        y_samples = np.sqrt(2.0) * sigma_y * nodes
        prefactor = 1.0 / np.sqrt(np.pi)

        slice_error_rates = np.zeros(self.m)

        for stage, bit_idx in enumerate(bit_indices):
            previous_indices = bit_indices[:stage]

            prefixes: dict[tuple[str, ...], list[int]] = {}
            for idx in range(self.number_of_intervals):
                key = tuple(bit_strings[idx][prev_idx] for prev_idx in previous_indices)
                prefixes.setdefault(key, []).append(idx)

            e_k = 0.0
            for idxs in prefixes.values():
                g0 = np.zeros_like(y_samples)
                g1 = np.zeros_like(y_samples)

                for interval_index in idxs:
                    bit = bit_strings[interval_index][bit_idx]
                    lower = (x_tau_arr[interval_index] - mu_factor * y_samples) / sigma_cond
                    upper = (x_tau_arr[interval_index + 1] - mu_factor * y_samples) / sigma_cond
                    contrib = ndtr(upper) - ndtr(lower)
                    if bit == "0":
                        g0 += contrib
                    else:
                        g1 += contrib

                error_density = np.minimum(g0, g1)
                e_k += prefactor * np.dot(weights, error_density)

            slice_error_rates[stage] = e_k

        return slice_error_rates

    def _leak_from_error_rate(self, *, slice_error_rates, coding_efficiencies=None, **_kwargs) -> float:
        if slice_error_rates is None:
            raise ValueError(
                "slice_error_rates must be provided when using the error-rate leak strategy."
            )

        errors = np.asarray(slice_error_rates, dtype=float)
        effective_errors = np.array([self._effective_bsc_error(e) for e in errors], dtype=float)
        capacities = np.array([self._bsc_capacity(e) for e in effective_errors], dtype=float)

        if coding_efficiencies is None:
            coding_efficiencies = np.ones_like(capacities)
        else:
            coding_efficiencies = np.asarray(coding_efficiencies, dtype=float)
            if coding_efficiencies.shape != capacities.shape:
                raise ValueError("coding_efficiencies must have the same shape as slice_error_rates")

        leak_terms = np.clip(1.0 - coding_efficiencies * capacities, 0.0, 1.0)
        return float(leak_terms.sum())


    def _leak_from_slepian_wolf(self, *, tau_arr, quadrature_points: int = 80, **_kwargs) -> float:
        """Compute the reconciliation leak via the Slepian-Wolf bound."""
        tau_arr = np.asarray(tau_arr, dtype=float)
        x_tau = np.sqrt(self.modulation_variance) * tau_arr

        var_x = self.modulation_variance
        var_y = self.bob_variance
        cov_xy = np.sqrt(self.transmittance / 2.0) * self.modulation_variance

        cond_var = max(var_x - (cov_xy ** 2) / var_y, 1e-12)
        cond_sigma = np.sqrt(cond_var)

        nodes, weights = _cached_hermgauss(quadrature_points)

        finite_edges = x_tau[np.isfinite(x_tau)]
        is_symmetric = finite_edges.size == 0 or np.allclose(finite_edges, -finite_edges[::-1])
        if is_symmetric:
            positive_mask = nodes > 0.0
            zero_mask = np.isclose(nodes, 0.0)
            nodes_use = np.concatenate((nodes[positive_mask], nodes[zero_mask]))
            weights_use = np.concatenate((2.0 * weights[positive_mask], weights[zero_mask]))
        else:
            nodes_use = nodes
            weights_use = weights

        sigma_y = np.sqrt(var_y)
        y_samples = np.sqrt(2.0) * sigma_y * nodes_use
        mu = (cov_xy / var_y) * y_samples

        z = (x_tau[np.newaxis, :] - mu[:, np.newaxis]) / cond_sigma
        cdf_values = ndtr(z)
        interval_probs = np.diff(cdf_values, axis=1)
        interval_probs = np.clip(interval_probs, 0.0, 1.0)

        totals = interval_probs.sum(axis=1, keepdims=True)
        norm_probs = np.zeros_like(interval_probs)
        valid_mask = totals[:, 0] > 0.0
        if np.any(valid_mask):
            norm_probs[valid_mask] = interval_probs[valid_mask] / totals[valid_mask]

        entropy = np.zeros(nodes_use.shape[0], dtype=float)
        if np.any(valid_mask):
            valid_probs = norm_probs[valid_mask]
            log_probs = np.zeros_like(valid_probs)
            np.log2(valid_probs, out=log_probs, where=valid_probs > 0.0)
            entropy_values = -np.sum(valid_probs * log_probs, axis=1)
            entropy[valid_mask] = entropy_values

        prefactor = 1.0 / np.sqrt(np.pi)
        return prefactor * np.dot(weights_use, entropy)


    @staticmethod
    def _normal_cdf(x, mu, sigma):
        if np.isneginf(x):
            return 0.0
        if np.isposinf(x):
            return 1.0
        z = (x - mu) / (sigma * np.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))

    def evaluate_error_rate(self, *args, **kwargs):
        """TODO: provide a per-interval error rate if needed."""
        raise NotImplementedError
