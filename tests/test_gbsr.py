import numpy as np
import pytest
from scipy.stats import norm

from guard_band_postselection.GBSR import GBSR


@pytest.fixture()
def gbsr_instance():
    return GBSR(
        m=1,
        modulation_variance=2.0,
        transmittance=0.5,
        excess_noise=0.01,
        code_efficiency=0.95,
        holevo_strategy='conservative',
    )


def _symmetric_guard(guard_width: float):
    return [[0.0, 0.0], [guard_width, guard_width], [0.0, 0.0]]


def _expected_p_pass(instance: GBSR, guard_width: float) -> float:
    sigma_bob = np.sqrt(instance.bob_variance)
    guard_actual = guard_width * sigma_bob
    tail_mass = 1.0 - instance.py_rv.cdf(guard_actual)
    return 2.0 * tail_mass


def _expected_ps_moments(instance: GBSR, guard_width: float):
    rv = instance.Q_star_rv
    var_x = float(rv.cov[0, 0])
    var_y = float(rv.cov[1, 1])
    cov_xy = float(rv.cov[0, 1])

    sigma_bob = np.sqrt(instance.bob_variance)
    guard_actual = guard_width * sigma_bob
    sigma_y = np.sqrt(var_y)
    if sigma_y == 0.0:
        raise ValueError('Y variance must be positive')

    z = guard_actual / sigma_y
    phi = norm.pdf(z)
    cdf = norm.cdf(z)
    pass_mass = 2.0 * (1.0 - cdf)
    if pass_mass == 0.0:
        raise ValueError('Guard width removes all mass')

    second_moment = sigma_y ** 2 * 2.0 * (phi * z + (1.0 - cdf))
    expected_y2 = second_moment / pass_mass

    conditional_var = var_x - (cov_xy ** 2) / var_y
    expected_x2 = conditional_var + ((cov_xy ** 2) / (var_y ** 2)) * expected_y2
    expected_xy = (cov_xy / var_y) * expected_y2
    return expected_x2, expected_y2, expected_xy, pass_mass


def _expected_q_star_pass(instance: GBSR, guard_width: float) -> float:
    sigma_bob = np.sqrt(instance.bob_variance)
    guard_actual = guard_width * sigma_bob
    sigma_y = np.sqrt(instance.Q_star_rv.cov[1, 1])
    z = guard_actual / sigma_y
    return 2.0 * (1.0 - norm.cdf(z))


@pytest.mark.parametrize('guard_width', [0.0, 0.2, 0.7])
def test_p_pass_matches_gaussian_cdf(gbsr_instance, guard_width):
    tau = [-np.inf, 0.0, np.inf]
    g = _symmetric_guard(guard_width)
    measured = gbsr_instance.evaluate_p_pass(tau, g)
    expected = _expected_p_pass(gbsr_instance, guard_width)
    assert measured == pytest.approx(expected, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize('guard_width', [0.0, 0.25, 0.75])
def test_Q_PS_moments_match_truncated_gaussian(gbsr_instance, guard_width):
    tau = [-np.inf, 0.0, np.inf]
    g = _symmetric_guard(guard_width)
    result = gbsr_instance._compute_Q_PS_moments(gbsr_instance.Q_star_rv, tau, g)
    exp_x2, exp_y2, exp_xy, _ = _expected_ps_moments(gbsr_instance, guard_width)
    assert result[0] == pytest.approx(exp_x2, rel=3e-6, abs=1e-8)
    assert result[1] == pytest.approx(exp_y2, rel=3e-6, abs=1e-8)
    assert result[2] == pytest.approx(exp_xy, rel=3e-6, abs=1e-8)


@pytest.mark.parametrize('guard_width', [0.0, 0.4, 0.9])
def test_integrate_Q_PS_matches_reference_ratios(gbsr_instance, guard_width):
    tau = [-np.inf, 0.0, np.inf]
    g = _symmetric_guard(guard_width)
    p_pass = gbsr_instance.evaluate_p_pass(tau, g)

    total = gbsr_instance._integrate_Q_PS([-np.inf, np.inf], [-np.inf, np.inf], tau, g)
    q_pass = _expected_q_star_pass(gbsr_instance, guard_width)
    assert total == pytest.approx(q_pass / p_pass, rel=1e-6, abs=1e-8)

    sigma_bob = np.sqrt(gbsr_instance.bob_variance)
    guard_actual = guard_width * sigma_bob
    positive = gbsr_instance._integrate_Q_PS([-np.inf, np.inf], [guard_actual, np.inf], tau, g)
    sigma_y = np.sqrt(gbsr_instance.Q_star_rv.cov[1, 1])
    numerator = norm.sf(guard_actual, loc=0.0, scale=sigma_y)
    assert positive == pytest.approx(numerator / p_pass, rel=1e-6, abs=1e-8)


def _stub_guard_band(instance, p_pass=0.73, error_rate=0.11):
    def fake_p_pass(*args, **kwargs):
        instance.p_pass = p_pass
        return p_pass

    instance.evaluate_p_pass = fake_p_pass
    instance.evaluate_error_rate = lambda *args, **kwargs: error_rate
    return p_pass, error_rate


def test_reconciliation_efficiency_consistency(gbsr_instance):
    tau = [-np.inf, 0.0, np.inf]
    g = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _stub_guard_band(gbsr_instance)
    metrics = gbsr_instance.evaluate_reconciliation_efficiency(tau, g, code_efficiency=0.95)
    numerator = metrics["bits_sent"] - metrics["bits_leaked"]
    assert metrics["eta"] * metrics["I_AB"] == pytest.approx(numerator, rel=1e-6)


def test_key_rate_matches_eta(gbsr_instance):
    tau = [-np.inf, 0.0, np.inf]
    g = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _stub_guard_band(gbsr_instance)
    metrics = gbsr_instance.evaluate_reconciliation_efficiency(tau, g, code_efficiency=0.95)
    expected_key_rate = (metrics["bits_sent"] - metrics["bits_leaked"]) - gbsr_instance._holevo_with_qct()
    assert metrics["key_rate"] == pytest.approx(expected_key_rate, rel=1e-6)


def test_key_rate_method_uses_eta(gbsr_instance):
    tau = [-np.inf, 0.0, np.inf]
    g = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _stub_guard_band(gbsr_instance)
    direct = gbsr_instance.evaluate_key_rate_in_bits_per_pulse(tau, g)
    metrics = gbsr_instance.evaluate_reconciliation_efficiency(tau, g)
    assert direct == pytest.approx(metrics["key_rate"], rel=1e-6)


def test_eta_decreases_with_lower_code_efficiency(gbsr_instance):
    tau = [-np.inf, 0.0, np.inf]
    g = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _stub_guard_band(gbsr_instance)
    hi = gbsr_instance.evaluate_reconciliation_efficiency(tau, g, code_efficiency=0.95)["eta"]
    lo = gbsr_instance.evaluate_reconciliation_efficiency(tau, g, code_efficiency=0.80)["eta"]
    assert lo < hi


def test_bits_leakage_matches_guard_band_formula(gbsr_instance):
    tau = [-np.inf, 0.0, np.inf]
    g = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _stub_guard_band(gbsr_instance)
    metrics = gbsr_instance.evaluate_reconciliation_efficiency(tau, g, code_efficiency=0.95)
    expected = gbsr_instance.m * metrics["leak_per_bit"] * metrics["p_pass"]
    assert metrics["bits_leaked"] == pytest.approx(expected, rel=1e-6)


def test_leak_per_bit_matches_eta_capacity(gbsr_instance):
    tau = [-np.inf, 0.0, np.inf]
    g = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _stub_guard_band(gbsr_instance)
    metrics = gbsr_instance.evaluate_reconciliation_efficiency(tau, g, code_efficiency=0.95)
    eta_c = float(metrics["coding_efficiency"])
    capacity = metrics["capacity"]
    expected = max(1.0 - eta_c * capacity, 0.0)
    assert metrics["leak_per_bit"] == pytest.approx(expected, rel=1e-6)
    assert expected == pytest.approx(1.0 - metrics["code_rate"], rel=1e-6)


def test_guard_band_leak_helper_matches_metrics(gbsr_instance):
    tau = [-np.inf, 0.0, np.inf]
    g = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _stub_guard_band(gbsr_instance)
    metrics = gbsr_instance.evaluate_reconciliation_efficiency(tau, g, code_efficiency=0.95)
    helper = gbsr_instance._leak_from_error_rate(
        error_rate=metrics["error_rate"], coding_efficiency=metrics["coding_efficiency"]
    )
    assert helper == pytest.approx(gbsr_instance.m * metrics["leak_per_bit"], rel=1e-6)
