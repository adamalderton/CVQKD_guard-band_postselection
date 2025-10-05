import numpy as np
import pytest

from guard_band_postselection.GBSR import GBSR


@pytest.fixture()
def gbsr_instance():
    return GBSR(
        m=1,
        modulation_variance=2.0,
        transmittance=0.5,
        excess_noise=0.01,
        code_efficiency=0.95,
        holevo_strategy="conservative",
    )




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
