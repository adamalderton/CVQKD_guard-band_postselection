"""Guard-band sliced reconciliation protocol."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import dblquad
from scipy.stats import norm
from tqdm import tqdm

from .key_efficiency_base import KeyEfficiencyBase


class GBSR(KeyEfficiencyBase):
    """Guard-band sliced reconciliation keyed on the shared base machinery."""

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
        holevo_strategy: str = "conservative",
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
        allowed_strategies = {"conservative", "optimistic"}
        strategy = holevo_strategy.lower()
        if strategy not in allowed_strategies:
            raise ValueError("holevo_strategy must be 'conservative' or 'optimistic'")
        self.holevo_strategy = strategy


        self.gaussian_attack_holevo_information = self._evaluate_holevo_information()
        self.devetak_winter = self.I_AB - self.gaussian_attack_holevo_information

    def plot_guard_band_diagram(self, normalised_tau_arr, normalised_g_arr) -> None:
        """Plot Bob's marginal together with the guard bands."""
        tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        fig, ax = plt.subplots(figsize=(2.5 * plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]))

        x = np.linspace(-2.5 * self.py_rv.var(), 2.5 * self.py_rv.var(), 200)
        y = self.py_rv.pdf(x)
        ax.plot(x, y, "r-", label="Bob's marginal")

        for i in range(len(tau_arr)):
            ax.plot([tau_arr[i], tau_arr[i]], [0, 1.1 * max(y)], "k-")
            ax.plot([tau_arr[i] - g_arr[i][0], tau_arr[i] - g_arr[i][0]], [0, 1.1 * max(y)], "k--")
            ax.plot([tau_arr[i] + g_arr[i][1], tau_arr[i] + g_arr[i][1]], [0, 1.1 * max(y)], "k--")

        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([0, 1.2 * max(y)])
        ax.legend()
        plt.show()

    def plot_Q_marginals(self, normalised_tau_arr, normalised_g_arr,
                          axis_range=(-10, 10), num_points_on_axis=100, add_originals=False) -> None:
        """Plot Husimi-Q marginals for the post-selected state."""
        self._evaluate_marginals(normalised_tau_arr, normalised_g_arr, axis_range, num_points_on_axis)

        axis_values = np.linspace(axis_range[0], axis_range[1], num_points_on_axis)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].plot(axis_values, self.px_Q_PS_values, "k-")
        axs[0].set_title("p(x)")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("Probability density")

        if add_originals:
            originals = self.px_rv.pdf(axis_values)
            originals /= np.sum(originals)
            axs[0].plot(axis_values, originals, "k--")

        axs[1].plot(axis_values, self.py_Q_PS_values, "k-")
        axs[1].set_title("p(y)")
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("Probability density")

        if add_originals:
            originals = self.py_rv.pdf(axis_values)
            originals /= np.sum(originals)
            axs[1].plot(axis_values, originals, "k--")

        plt.tight_layout()
        plt.show()

    def evaluate_key_rate_in_bits_per_pulse(self, tau_arr, g_arr) -> float:
        metrics = self.evaluate_reconciliation_efficiency(tau_arr, g_arr)
        return metrics["key_rate"]

    def evaluate_reconciliation_efficiency(
        self,
        tau_arr,
        g_arr,
        *,
        code_efficiency=None,
        bit_assignment="Gray",
    ) -> dict[str, float]:
        """Evaluate the guard-band reconciliation efficiency at fixed decoder speed."""
        quantisation_entropy = self.evaluate_quantisation_entropy(tau_arr)
        p_pass = self.evaluate_p_pass(tau_arr, g_arr)
        raw_error = self.evaluate_error_rate(tau_arr, g_arr, bit_assignment=bit_assignment)
        effective_error = self._effective_bsc_error(raw_error)
        capacity = self._bsc_capacity(effective_error)
        coding_efficiency = float(
            self._evaluate_code_efficiency(np.array([effective_error]), override=code_efficiency)[0]
        )
        code_rate = float(coding_efficiency * capacity)
        leak_per_bit = max(1.0 - code_rate, 0.0)
        bits_sent = p_pass * quantisation_entropy
        bits_leaked = p_pass * self.m * leak_per_bit
        numerator = bits_sent - bits_leaked
        denominator = max(self.I_AB, 1e-12)
        eta = numerator / denominator if denominator > 0.0 else 0.0
        if self.holevo_strategy == "optimistic":
            self.evaluate_cov_mat_PS(tau_arr, g_arr)
            holevo_information = self._evaluate_holevo_information(
                self.a_PS, self.b_PS, self.c_PS
            )
            key_rate = numerator - (holevo_information + self.Delta_QCT)
        else:
            key_rate = numerator - self._holevo_with_qct()
        return {
            "eta": float(eta),
            "bits_sent": float(bits_sent),
            "bits_leaked": float(bits_leaked),
            "key_rate": float(key_rate),
            "p_pass": float(p_pass),
            "raw_error_rate": float(raw_error),
            "error_rate": float(effective_error),
            "capacity": float(capacity),
            "coding_efficiency": coding_efficiency,
            "code_rate": float(code_rate),
            "leak_per_bit": float(leak_per_bit),
            "quantisation_entropy": float(quantisation_entropy),
            "I_AB": self.I_AB,
        }

    def evaluate_error_rate(self, normalised_tau_arr, normalised_g_arr, bit_assignment="Gray") -> float:
        x_tau_arr = np.sqrt(self.modulation_variance) * np.array(normalised_tau_arr)
        y_tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        y_g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        if bit_assignment == "Gray":
            bit_strings = self._generate_gray_bit_assignment(self.m)
        elif bit_assignment == "binary":
            bit_strings = self._generate_binary_bit_assignment(self.m)
        else:
            raise ValueError("Invalid bit assignment scheme. Choose 'Gray' or 'binary'.")

        error_rate = 0.0
        for i in range(self.number_of_intervals):
            for j in range(self.number_of_intervals):
                if i == j:
                    continue

                normalised_hamming_distance = self._hamming_distance(bit_strings[i], bit_strings[j]) / self.m
                xlims = [x_tau_arr[i], x_tau_arr[i + 1]]
                ylims = [y_tau_arr[j] + y_g_arr[j][1], y_tau_arr[j + 1] - y_g_arr[j + 1][0]]

                error_rate += normalised_hamming_distance * self._integrate_2D_gaussian_pdf(
                    self.joint_rv, xlims, ylims
                )

        return error_rate

    def evaluate_p_pass(self, normalised_tau_arr, normalised_g_arr) -> float:
        tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        p_pass = 0.0
        for i in range(len(tau_arr) - 1):
            p_pass += self._integrate_1D_gaussian_pdf(
                self.py_rv,
                [tau_arr[i] + g_arr[i][1], tau_arr[i + 1] - g_arr[i + 1][0]],
            )

        self.p_pass = p_pass
        return self.p_pass

    def evaluate_cov_mat_PS(self, normalised_tau_arr, normalised_g_arr):
        var_x, var_y, cov_xy = self._compute_Q_PS_moments(
            self.Q_star_rv, normalised_tau_arr, normalised_g_arr
        )

        effective_husimi_cov_mat = np.zeros((2, 2))
        effective_husimi_cov_mat[0][0] = var_x
        effective_husimi_cov_mat[1][1] = var_y
        effective_husimi_cov_mat[0][1] = cov_xy
        effective_husimi_cov_mat[1][0] = cov_xy

        self.cov_mat_PS = effective_husimi_cov_mat - np.eye(2)
        self.a_PS = self.cov_mat_PS[0][0]
        self.b_PS = self.cov_mat_PS[1][1]
        self.c_PS = self.cov_mat_PS[0][1]
        return self.cov_mat_PS

    def evaluate_slepian_wolf_leakage(self, normalised_tau_arr, normalised_g_arr) -> float:
        """Numerical Slepian-Wolf leakage estimate (placeholder implementation)."""
        # TODO: integrate the notes formulation more efficiently and expose configuration hooks.
        num_y_points = 1000
        sigma_Y = np.sqrt(self.bob_variance)
        y_min = -5 * sigma_Y
        y_max = 5 * sigma_Y
        y_vals = np.linspace(y_min, y_max, num_y_points)
        delta_y = y_vals[1] - y_vals[0]

        p_y = np.array([self._compute_p_y(y_j) for y_j in y_vals])
        epsilon = 1e-12
        p_y = np.clip(p_y, epsilon, None)

        num_intervals = self.number_of_intervals
        p_s_y = np.zeros((num_intervals, num_y_points))

        x_tau_arr = np.sqrt(self.modulation_variance) * np.array(normalised_tau_arr)

        for i in range(num_intervals):
            xlims = [x_tau_arr[i], x_tau_arr[i + 1]]
            for idx, y_j in enumerate(y_vals):
                p_s_y[i, idx] = self._integrate_pxy_over_x(xlims, y_j)

        p_s_given_y = p_s_y / p_y

        H_S_given_y = np.zeros(num_y_points)
        for idx in range(num_y_points):
            p_s_given_y_j = np.clip(p_s_given_y[:, idx], epsilon, None)
            H_S_given_y[idx] = -np.sum(p_s_given_y_j * np.log2(p_s_given_y_j))

        H_S_given_Y = np.sum(H_S_given_y * p_y) * delta_y
        return H_S_given_Y

    def _evaluate_marginals(self, normalised_tau_arr, normalised_g_arr,
                            axis_range=(-10, 10), num_points_on_axis=100):
        tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        axis_values = np.linspace(axis_range[0], axis_range[1], num_points_on_axis)

        def filter_function(y):
            mask = np.ones_like(y, dtype=bool)
            for i in range(len(tau_arr)):
                y_minus_tau = y - tau_arr[i]
                condition = (-g_arr[i][0] <= y_minus_tau) & (y_minus_tau <= g_arr[i][1])
                mask &= ~condition
            return mask.astype(int)

        x_mesh, y_mesh = np.meshgrid(axis_values, axis_values)
        Q_star_values = self.Q_star_rv.pdf(np.dstack((x_mesh, y_mesh)))

        self.Q_PS_values = filter_function(y_mesh) * Q_star_values
        self.Q_PS_values /= np.sum(self.Q_PS_values)

        self.px_Q_PS_values = np.sum(self.Q_PS_values, axis=0)
        self.py_Q_PS_values = np.sum(self.Q_PS_values, axis=1)
        self.px_Q_PS_values /= np.sum(self.px_Q_PS_values)
        self.py_Q_PS_values /= np.sum(self.py_Q_PS_values)
        return self.px_Q_PS_values, self.py_Q_PS_values, self.Q_PS_values

    def _compute_Q_PS_moments(self, rv, normalised_tau_arr, normalised_g_arr):
        y_tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        y_g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        epsabs = 1e-3
        epsrel = 1e-3

        def integrand_x2(y, x):
            return x ** 2 * rv.pdf([x, y])

        def integrand_y2(y, x):
            return y ** 2 * rv.pdf([x, y])

        def integrand_xy(y, x):
            return x * y * rv.pdf([x, y])

        xlims = [-np.inf, np.inf]
        ylims = [-np.inf, np.inf]

        band_ranges = []
        for i in range(len(y_tau_arr)):
            band_lower = y_tau_arr[i] - y_g_arr[i][0]
            band_upper = y_tau_arr[i] + y_g_arr[i][1]
            band_ranges.append((band_lower, band_upper))

        interval_list = [[ylims[0], ylims[1]]]
        for band_lower, band_upper in band_ranges:
            new_intervals = []
            for start, end in interval_list:
                if band_upper <= start or band_lower >= end:
                    new_intervals.append([start, end])
                    continue
                if band_lower > start:
                    new_intervals.append([start, band_lower])
                if band_upper < end:
                    new_intervals.append([band_upper, end])
            interval_list = new_intervals

        if self.progress_bar:
            interval_iter = tqdm(interval_list)
        else:
            interval_iter = interval_list

        E_X2_num = 0.0
        E_Y2_num = 0.0
        E_XY_num = 0.0
        norm_const = 0.0

        for y_start, y_end in interval_iter:
            if y_end <= y_start:
                continue

            norm_const += self._integrate_2D_gaussian_pdf(rv, xlims, [y_start, y_end])
            E_X2_num += dblquad(integrand_x2, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end, epsabs=epsabs, epsrel=epsrel)[0]
            E_Y2_num += dblquad(integrand_y2, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end, epsabs=epsabs, epsrel=epsrel)[0]
            E_XY_num += dblquad(integrand_xy, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end, epsabs=epsabs, epsrel=epsrel)[0]

        if norm_const == 0.0:
            raise ValueError("Normalization constant is zero. Check integration limits and guard bands.")

        var_x = E_X2_num / norm_const
        var_y = E_Y2_num / norm_const
        cov_xy = E_XY_num / norm_const
        return var_x, var_y, cov_xy

    def _integrate_Q_PS(self, xlims, ylims, normalised_tau_arr, normalised_g_arr):
        y_tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        y_g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        integral_result = 0.0
        y_lower = ylims[0]
        y_upper = ylims[1]

        band_ranges = []
        for i in range(len(y_tau_arr)):
            band_lower = y_tau_arr[i] - y_g_arr[i][0]
            band_upper = y_tau_arr[i] + y_g_arr[i][1]
            band_ranges.append((band_lower, band_upper))

        interval_list = [[y_lower, y_upper]]
        for band_lower, band_upper in band_ranges:
            new_intervals = []
            for start, end in interval_list:
                if band_upper <= start or band_lower >= end:
                    new_intervals.append([start, end])
                    continue
                if band_lower > start:
                    new_intervals.append([start, band_lower])
                if band_upper < end:
                    new_intervals.append([band_upper, end])
            interval_list = new_intervals

        for start, end in interval_list:
            if end > start:
                integral_result += (1.0 / self.p_pass) * self._integrate_2D_gaussian_pdf(
                    self.Q_star_rv, xlims, [start, end]
                )

        return integral_result

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
        leak_per_bit = np.clip(1.0 - coding_efficiency * capacity, 0.0, 1.0)
        return float(self.m * leak_per_bit)

    def _leak_from_slepian_wolf(self, **_kwargs) -> float:
        raise NotImplementedError(
            "Slepian-Wolf leakage computation is not implemented for GBSR yet."
        )

    def _compute_p_y(self, y_val: float) -> float:
        return self.py_rv.pdf(y_val)

    def _integrate_pxy_over_x(self, xlims, y_val: float) -> float:
        """Integrate p(x, y) over the interval xlims for a fixed y value."""
        var_y = self.cov_mat[1, 1]
        cov_xy = self.cov_mat[0, 1]
        var_x = self.cov_mat[0, 0]

        conditional_mean = (cov_xy / var_y) * y_val
        conditional_var = var_x - (cov_xy ** 2) / var_y
        conditional_sigma = np.sqrt(max(conditional_var, 1e-12))
        conditional = norm(loc=conditional_mean, scale=conditional_sigma)
        probability = conditional.cdf(xlims[1]) - conditional.cdf(xlims[0])
        return probability * self._compute_p_y(y_val)

    def _evaluate_actual_mutual_information(self, normalised_tau_arr, normalised_g_arr):
        """TODO: implement the actual mutual information calculation."""
        raise NotImplementedError

    def _evaluate_postselected_devetak_winter(self, normalised_tau_arr, normalised_g_arr):
        """TODO: implement the post-selected Devetak-Winter bound."""
        return self._evaluate_actual_mutual_information(normalised_tau_arr, normalised_g_arr) - self.gaussian_attack_holevo_information





