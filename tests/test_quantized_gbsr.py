import types

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from guard_band_postselection import QuantizedGBSR
from guard_band_postselection.quantized_gbsr import _rect_prob


def test_rect_prob_matches_independent_case():
    lower = 0.0
    upper = 1.0
    expected = (norm.cdf(upper) - norm.cdf(lower)) ** 2
    result = _rect_prob(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, lower, upper, lower, upper)
    assert result == pytest.approx(expected, rel=1e-6)


def test_gbsr_init_builds_channel_statistics():
    gbsr = QuantizedGBSR(m=1, modulation_variance=1.0, transmittance=0.1, excess_noise=0.0)
    assert gbsr.cov_mat.shape == (2, 2)
    assert np.isfinite(gbsr.I_AB)


@pytest.fixture
def bare_gbsr():
    obj = object.__new__(QuantizedGBSR)
    obj.modulation_variance = 1.0
    obj.transmittance = 0.4
    obj._constant_excess_noise = 0.2
    obj.shot_noise = 1.0
    obj.m = 1
    obj.number_of_intervals = 2
    obj.symbols_per_pulse = 2.0
    obj.excess_noise = 0.2
    return obj


def test_refresh_channel_parameters_sets_expected_attributes(bare_gbsr):
    bare_gbsr.refresh_channel_parameters()
    assert bare_gbsr.bob_variance == pytest.approx(
        0.5 * bare_gbsr.transmittance * bare_gbsr.modulation_variance
        + bare_gbsr.shot_noise
        + 0.5 * bare_gbsr.excess_noise
    )
    assert bare_gbsr.SNR == pytest.approx(
        0.5
        * bare_gbsr.transmittance
        * bare_gbsr.modulation_variance
        / (bare_gbsr.shot_noise + 0.5 * bare_gbsr.excess_noise)
    )
    assert bare_gbsr.I_AB == pytest.approx(np.log2(1.0 + bare_gbsr.SNR))
    assert bare_gbsr.cov_mat_EB.shape == (2, 2)
    assert bare_gbsr.cov_mat_EB[0, 1] == pytest.approx(bare_gbsr.c)


def test_evaluate_quantisation_entropy_balanced_slices(bare_gbsr):
    bare_gbsr.px_rv = norm(loc=0.0, scale=np.sqrt(bare_gbsr.modulation_variance))
    normalised_tau = np.array([-np.inf, 0.0, np.inf])
    entropy = bare_gbsr.evaluate_quantisation_entropy(normalised_tau)
    assert entropy == pytest.approx(1.0, abs=1e-6)


def test_evaluate_quantisation_entropy_with_guard_uses_accepted_distribution(bare_gbsr):
    bare_gbsr.symbol_joint_probabilities = np.full((2, 2), 0.25)
    bare_gbsr.symbol_marginal_alice = np.array([0.25, 0.75])
    guards = np.zeros((3, 2))
    tau = np.array([-np.inf, 0.0, np.inf])
    entropy = bare_gbsr.evaluate_quantisation_entropy(tau, guards)
    expected = -(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))
    assert entropy == pytest.approx(expected, rel=1e-9)


def test_evaluate_symbol_mutual_information_independent_variables(bare_gbsr):
    bare_gbsr.modulation_variance = 1.0
    bare_gbsr.transmittance = 0.0
    bare_gbsr._constant_excess_noise = 0.0
    normalised_tau = np.array([-np.inf, 0.0, np.inf])
    normalised_g = np.zeros((3, 2))
    mi = bare_gbsr.evaluate_symbol_mutual_information(normalised_tau, normalised_g)
    assert mi == pytest.approx(0.0, abs=1e-3)
    assert bare_gbsr.p_pass == pytest.approx(1.0, abs=1e-6)


def test_evaluate_continuous_gaussian_holevo_matches_formula(bare_gbsr):
    a, b, c = 2.0, 3.0, 1.0
    expected = bare_gbsr._g(0.5 * (np.sqrt((a + b) ** 2 - 4 * c**2) + (b - a)))
    expected += bare_gbsr._g(0.5 * (np.sqrt((a + b) ** 2 - 4 * c**2) - (b - a)))
    expected -= bare_gbsr._g(a - c**2 / (b + 1.0))
    assert bare_gbsr._evaluate_continuous_Gaussian_holevo(a, b, c) == pytest.approx(
        expected, rel=1e-9
    )


def test_g_returns_zero_below_one(bare_gbsr):
    assert bare_gbsr._g(1.0) == 0.0
    assert bare_gbsr._g(5.0) > 0.0


def test_binary_entropy_valid_and_invalid(bare_gbsr):
    assert bare_gbsr._binary_entropy(0.25) == pytest.approx(0.811278, rel=1e-6)
    with pytest.raises(ValueError):
        bare_gbsr._binary_entropy(1.5)


def test_integrate_1d_gaussian_pdf_interval(bare_gbsr):
    rv = norm()
    result = bare_gbsr._integrate_1D_gaussian_pdf(rv, [-1.0, 1.0])
    assert result == pytest.approx(norm.cdf(1.0) - norm.cdf(-1.0), rel=1e-9)


def test_integrate_2d_gaussian_pdf_fullspace(bare_gbsr):
    rv = types.SimpleNamespace(mean=np.zeros(2), cov=np.eye(2))
    assert bare_gbsr._integrate_2D_gaussian_pdf(
        rv, [-np.inf, np.inf], [-np.inf, np.inf]
    ) == pytest.approx(1.0)


def test_integrate_2d_gaussian_pdf_rect(bare_gbsr):
    mv = multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
    result = bare_gbsr._integrate_2D_gaussian_pdf(mv, [0.0, 1.0], [0.0, 1.0])
    expected = (norm.cdf(1.0) - norm.cdf(0.0)) ** 2
    assert result == pytest.approx(expected, rel=1e-6)


def test_channel_gaussian_parameters_values(bare_gbsr):
    bare_gbsr.modulation_variance = 1.0
    bare_gbsr.transmittance = 0.5
    bare_gbsr._constant_excess_noise = 0.1
    a, v_y, c = bare_gbsr._channel_gaussian_parameters()
    assert a == pytest.approx(bare_gbsr.modulation_variance + 1.0)
    assert v_y > 0.0
    assert c == pytest.approx(
        np.sqrt(bare_gbsr.transmittance) * np.sqrt(max(a * a - 1.0, 0.0))
    )


def test_safe_lower_bound_infinite_cases():
    assert QuantizedGBSR._safe_lower_bound(-np.inf, 1.0) == -np.inf
    assert np.isposinf(QuantizedGBSR._safe_lower_bound(1.0, np.inf))
    assert QuantizedGBSR._safe_lower_bound(1.0, 2.0) == 3.0


def test_safe_upper_bound_infinite_cases():
    assert QuantizedGBSR._safe_upper_bound(np.inf, 1.0) == np.inf
    assert np.isneginf(QuantizedGBSR._safe_upper_bound(1.0, np.inf))
    assert QuantizedGBSR._safe_upper_bound(3.0, 1.0) == 2.0


def test_truncated_standard_normal_moments():
    mass, first, second = QuantizedGBSR._truncated_standard_normal_moments(0.0, 1.0)
    expected_mass = norm.cdf(1.0) - norm.cdf(0.0)
    expected_first = norm.pdf(0.0) - norm.pdf(1.0)
    expected_second = expected_mass - (1.0 * norm.pdf(1.0) - 0.0 * norm.pdf(0.0))
    assert mass == pytest.approx(expected_mass, rel=1e-9)
    assert first == pytest.approx(expected_first, rel=1e-9)
    assert second == pytest.approx(expected_second, rel=1e-9)


def test_build_acceptance_statistics_two_intervals(bare_gbsr):
    sigma_y = 1.0
    tau = np.array([-np.inf, 0.0, np.inf])
    guard = np.zeros((3, 2))
    bounds, std_bounds, masses, firsts, seconds = bare_gbsr._build_acceptance_statistics(
        tau, guard, sigma_y
    )
    assert bounds == [(-np.inf, 0.0), (0.0, np.inf)]
    assert std_bounds == [(-np.inf, 0.0), (0.0, np.inf)]
    assert np.allclose(masses, [0.5, 0.5])
    assert np.allclose(firsts, [-norm.pdf(0.0), norm.pdf(0.0)])
    assert np.allclose(seconds, [0.5, 0.5])


def test_entangling_cloner_matrices_shapes(bare_gbsr):
    Sigma_B, Sigma_E, Sigma_EB, V, v_y, b = bare_gbsr._entangling_cloner_matrices()
    assert Sigma_B.shape == (2, 2)
    assert Sigma_E.shape == (4, 4)
    assert Sigma_EB.shape == (4, 2)
    assert V > 0.0 and v_y > 0.0 and b > 0.0


def test_symplectic_eigenvalues_two_mode():
    cov = np.diag([2.0, 2.0, 3.0, 3.0])
    eigenvalues = QuantizedGBSR._symplectic_eigenvalues_two_mode(cov)
    assert np.allclose(np.sort(eigenvalues), [2.0, 3.0])


def test_generate_gray_bit_assignment(bare_gbsr):
    assignment = bare_gbsr._generate_gray_bit_assignment(2)
    assert assignment == ["00", "01", "11", "10"]


def test_generate_binary_bit_assignment(bare_gbsr):
    assignment = bare_gbsr._generate_binary_bit_assignment(2)
    assert assignment == ["00", "01", "10", "11"]


def test_hamming_distance(bare_gbsr):
    assert bare_gbsr._hamming_distance("0101", "0111") == 1


def test_plot_guard_band_diagram_calls_expected(monkeypatch, bare_gbsr):
    class DummyAx:
        def __init__(self):
            self.plot_calls = 0
            self.axvspan_calls = 0

        def plot(self, *args, **kwargs):
            self.plot_calls += 1

        def axvspan(self, *args, **kwargs):
            self.axvspan_calls += 1

        def set_xlim(self, *args, **kwargs):
            pass

        def set_ylim(self, *args, **kwargs):
            pass

        def set_xlabel(self, *args, **kwargs):
            pass

        def set_ylabel(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            pass

    dummy_ax = DummyAx()

    def fake_subplots(*args, **kwargs):
        return object(), dummy_ax

    bare_gbsr.bob_variance = 1.0
    bare_gbsr.py_rv = norm()

    monkeypatch.setattr("guard_band_postselection.quantized_gbsr.plt.subplots", fake_subplots)
    monkeypatch.setattr("guard_band_postselection.quantized_gbsr.plt.show", lambda: None)

    tau = np.array([-np.inf, 0.0, np.inf])
    guards = np.zeros((3, 2))
    bare_gbsr.plot_guard_band_diagram(tau, guards)

    assert dummy_ax.plot_calls >= 2
    assert dummy_ax.axvspan_calls == len(tau)


def test_evaluate_quantised_holevo_information_with_zero_coupling(monkeypatch, bare_gbsr):
    bare_gbsr.number_of_intervals = 2

    def fake_cov_beta_acc(self, tau, guards):
        return np.eye(2)

    def fake_entangling():
        return np.eye(2), 2.0 * np.eye(4), np.zeros((4, 2)), 1.0, 1.0, 1.0

    def fake_symplectic(_):
        return np.array([1.2, 1.3])

    bare_gbsr.evaluate_cov_beta_acc = types.MethodType(fake_cov_beta_acc, bare_gbsr)
    bare_gbsr._entangling_cloner_matrices = types.MethodType(
        lambda self: fake_entangling(), bare_gbsr
    )
    bare_gbsr._symplectic_eigenvalues_two_mode = types.MethodType(
        lambda self, matrix: fake_symplectic(matrix), bare_gbsr
    )
    bare_gbsr._g = types.MethodType(lambda self, value: value, bare_gbsr)

    tau = np.array([-np.inf, 0.0, np.inf])
    guards = np.zeros((3, 2))

    holevo = bare_gbsr.evaluate_quantised_holevo_information(tau, guards)
    assert holevo == pytest.approx(0.0)
    assert np.allclose(bare_gbsr.sigma_E_conditioned, 2.0 * np.eye(4))
    assert np.allclose(bare_gbsr.sigma_E_accepted, 2.0 * np.eye(4))


def test_evaluate_quantised_maximum_key_efficiency_uses_metrics(bare_gbsr):
    bare_gbsr.number_of_intervals = 2
    bare_gbsr.symbols_per_pulse = 2.0
    bare_gbsr.p_pass = 0.4
    bare_gbsr.symbol_joint_probabilities = np.full((2, 2), 0.25)
    bare_gbsr.symbol_marginal_alice = np.array([0.5, 0.5])
    bare_gbsr.symbol_marginal_bob = np.array([0.5, 0.5])

    bare_gbsr.evaluate_symbol_mutual_information = types.MethodType(
        lambda self, tau, guards: 0.8, bare_gbsr
    )
    bare_gbsr.evaluate_quantised_holevo_information = types.MethodType(
        lambda self, tau, guards: 0.3, bare_gbsr
    )

    tau = np.array([-np.inf, 0.0, np.inf])
    guards = np.zeros((3, 2))

    metrics = bare_gbsr.evaluate_quantised_maximum_key_efficiency(tau, guards)
    assert metrics["p_pass"] == pytest.approx(0.4)
    assert metrics["I_symbol"] == pytest.approx(0.8)
    assert metrics["holevo_bound"] == pytest.approx(0.3)
    assert metrics["H_Tx_acc"] == pytest.approx(1.0)
    assert metrics["H_Tx_given_Ty"] == pytest.approx(1.0)
    assert metrics["key_per_accepted_symbol"] == pytest.approx(0.5)
    assert metrics["key_per_pulse"] == pytest.approx(0.4)


def test_evaluate_practical_key_efficiency_raises_name_error(bare_gbsr):
    with pytest.raises(NameError):
        bare_gbsr.evaluate_practical_key_efficiency()


def test_leak_from_error_rate_uses_binary_entropy(bare_gbsr):
    bare_gbsr.m = 3
    leak = bare_gbsr._leak_from_error_rate(0.2)
    expected = 3 * bare_gbsr._binary_entropy(0.2)
    assert leak == pytest.approx(expected, rel=1e-9)


def test_evaluate_error_rate_with_stubbed_integral(bare_gbsr):
    bare_gbsr.modulation_variance = 1.0
    bare_gbsr.bob_variance = 1.0
    bare_gbsr.m = 1
    bare_gbsr.number_of_intervals = 2
    bare_gbsr.joint_rv = object()
    bare_gbsr._integrate_2D_gaussian_pdf = types.MethodType(
        lambda self, rv, xlims, ylims: 0.05, bare_gbsr
    )

    tau = np.array([-1.0, 0.0, 1.0])
    guards = np.zeros((3, 2))

    error_rate = bare_gbsr.evaluate_error_rate(tau, guards, bit_assignment="Gray")
    assert error_rate == pytest.approx(0.1)


def test_evaluate_slepian_wolf_leakage_returns_none(bare_gbsr):
    assert bare_gbsr.evaluate_slepian_wolf_leakage() is None


def test_evaluate_slepian_wolf_leakage_from_joint(bare_gbsr):
    bare_gbsr.symbol_joint_probabilities = np.full((2, 2), 0.25)
    bare_gbsr.symbol_marginal_bob = np.array([0.5, 0.5])
    guards = np.zeros((3, 2))
    tau = np.array([-np.inf, 0.0, np.inf])
    leak = bare_gbsr.evaluate_slepian_wolf_leakage(tau, guards)
    assert leak == pytest.approx(1.0, abs=1e-9)


def test_evaluate_p_pass_full_acceptance(bare_gbsr):
    bare_gbsr.bob_variance = 1.0
    bare_gbsr.py_rv = norm()
    tau = np.array([-np.inf, 0.0, np.inf])
    guards = np.zeros((3, 2))
    p_pass = bare_gbsr.evaluate_p_pass(tau, guards)
    assert p_pass == pytest.approx(1.0, abs=1e-6)
    assert bare_gbsr.p_pass == pytest.approx(1.0, abs=1e-6)


def test_evaluate_cov_beta_acc_no_guard(bare_gbsr):
    bare_gbsr.modulation_variance = 1.0
    bare_gbsr.transmittance = 0.0
    bare_gbsr._constant_excess_noise = 0.0
    bare_gbsr.number_of_intervals = 2
    tau = np.array([-np.inf, 0.0, np.inf])
    guards = np.zeros((3, 2))
    cov = bare_gbsr.evaluate_cov_beta_acc(tau, guards)
    assert np.allclose(cov, np.array([[2.0, 0.0], [0.0, 2.0]]))
    assert bare_gbsr.p_pass == pytest.approx(1.0, abs=1e-6)
    assert len(bare_gbsr.accepted_interval_bounds) == bare_gbsr.number_of_intervals
