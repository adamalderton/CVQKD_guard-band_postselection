import numpy as np
import pytest

from guard_band_postselection.MD import MD


@pytest.fixture()
def md_instance():
    return MD(
        modulation_variance=4.0,
        transmittance=0.6,
        excess_noise=0.05,
        code_efficiency=0.95,
    )


def test_biawgn_capacity_bounds(md_instance):
    assert md_instance._biawgn_capacity(0.0) == pytest.approx(0.0)
    high_capacity = md_instance._biawgn_capacity(5.0)
    assert 0.9 < high_capacity < 1.0


def test_capacity_monotonicity():
    md = MD(
        modulation_variance=3.0,
        transmittance=0.3,
        excess_noise=0.1,
        code_efficiency=0.95,
    )
    s_values = np.linspace(0.0, 3.0, 6)
    capacities = [md._biawgn_capacity(s) for s in s_values]
    assert capacities[0] == pytest.approx(0.0)
    assert all(capacities[i] <= capacities[i + 1] + 1e-9 for i in range(len(capacities) - 1))


def test_eta_consistency(md_instance):
    metrics = md_instance.evaluate_reconciliation_efficiency(code_efficiency=0.95)
    numerator = metrics["bits_sent"] - metrics["bits_leaked"]
    assert metrics["eta"] * metrics["I_AB"] == pytest.approx(numerator, rel=1e-6)


def test_key_rate_matches_eta(md_instance):
    metrics = md_instance.evaluate_reconciliation_efficiency(code_efficiency=0.95)
    expected_key_rate = (metrics["bits_sent"] - metrics["bits_leaked"]) - md_instance._holevo_with_qct()
    assert metrics["key_rate"] == pytest.approx(expected_key_rate, rel=1e-6)


def test_key_rate_method_uses_eta(md_instance):
    direct = md_instance.evaluate_key_rate_in_bits_per_pulse(code_efficiency=0.95)
    metrics = md_instance.evaluate_reconciliation_efficiency(code_efficiency=0.95)
    assert direct == pytest.approx(metrics["key_rate"], rel=1e-6)


def test_bits_leakage_matches_notes(md_instance):
    metrics = md_instance.evaluate_reconciliation_efficiency(code_efficiency=0.95)
    assert metrics["bits_sent"] == pytest.approx(1.0)
    expected = max(1.0 - metrics["code_rate"], 0.0)
    assert metrics["bits_leaked"] == pytest.approx(expected, rel=1e-6)
    eta_c = metrics["coding_efficiency"]
    capacity = metrics["capacity"]
    expected_formula = max(1.0 - eta_c * capacity, 0.0)
    assert metrics["leak_per_bit"] == pytest.approx(expected_formula, rel=1e-6)
    assert expected == pytest.approx(expected_formula, rel=1e-6)

def test_md_leak_helper_matches_metrics(md_instance):
    metrics = md_instance.evaluate_reconciliation_efficiency(code_efficiency=0.95)
    helper = md_instance._leak_from_error_rate(
        error_rate=metrics["error_rate"], coding_efficiency=metrics["coding_efficiency"]
    )
    assert helper == pytest.approx(metrics["bits_leaked"], rel=1e-6)

