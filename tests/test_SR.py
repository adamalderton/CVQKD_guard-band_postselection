import math
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

from guard_band_postselection.SR import SR


@pytest.fixture()
def sr_instance():
    return SR(
        m=1,
        modulation_variance=2.0,
        transmittance=0.5,
        excess_noise=0.01,
        code_efficiency=0.95,
    )


def test_evaluate_slice_error_rates(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    errors = sr_instance._evaluate_slice_error_rates(tau)
    assert errors.shape == (sr_instance.m,)
    assert np.all(errors >= 0.0)


def test_leak_from_error_rate(sr_instance):
    leak = sr_instance._leak_from_error_rate(slice_error_rates=[0.1])
    expected = sr_instance._binary_entropy(0.1)
    assert leak == pytest.approx(expected)


def test_normal_cdf_edges():
    assert SR._normal_cdf(float('-inf'), 0.0, 1.0) == pytest.approx(0.0)
    assert SR._normal_cdf(float('inf'), 0.0, 1.0) == pytest.approx(1.0)
    assert SR._normal_cdf(0.0, 0.0, 1.0) == pytest.approx(0.5)


def reference_slepian_wolf(sr: SR, tau_arr, quad_points=200):
    tau_arr = np.asarray(tau_arr, dtype=float)
    var_x = sr.modulation_variance
    var_y = sr.bob_variance
    cov_xy = np.sqrt(sr.transmittance / 2.0) * sr.modulation_variance
    cond_var = max(var_x - (cov_xy ** 2) / var_y, 1e-12)
    cond_sigma = np.sqrt(cond_var)

    sigma_y = np.sqrt(var_y)
    nodes, weights = hermgauss(quad_points)
    y_samples = np.sqrt(2.0) * sigma_y * nodes
    prefactor = 1.0 / math.sqrt(math.pi)

    conditional_entropy = 0.0
    for y, w in zip(y_samples, weights):
        mu = (cov_xy / var_y) * y
        probs = []
        for lower, upper in zip(tau_arr[:-1], tau_arr[1:]):
            lower_x = (lower - mu) / cond_sigma
            upper_x = (upper - mu) / cond_sigma
            probs.append(norm.cdf(upper_x) - norm.cdf(lower_x))
        probs = np.clip(probs, 0.0, 1.0)
        total = probs.sum()
        if total <= 0.0:
            continue
        probs /= total
        mask = probs > 0.0
        if not np.any(mask):
            continue
        h_cond = -np.sum(probs[mask] * np.log2(probs[mask]))
        conditional_entropy += w * h_cond

    return prefactor * conditional_entropy


def test_leak_from_slepian_wolf_matches_reference(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    reference = reference_slepian_wolf(sr_instance, tau, quad_points=200)
    leak = sr_instance._leak_from_slepian_wolf(tau_arr=tau, quadrature_points=80)
    assert leak == pytest.approx(reference, rel=1e-3)


def test_leak_from_slepian_wolf_independent_case():
    sr = SR(
        m=1,
        modulation_variance=1.5,
        transmittance=0.0,
        excess_noise=0.01,
        leak_strategy="slepian-wolf",
    )
    tau = [-np.inf, 0.0, np.inf]
    leak = sr._leak_from_slepian_wolf(tau_arr=tau, quadrature_points=60)
    assert leak == pytest.approx(1.0, rel=1e-3)


def test_leak_from_slepian_wolf_positive_correlation():
    sr = SR(
        m=1,
        modulation_variance=4.0,
        transmittance=0.8,
        excess_noise=0.0,
        leak_strategy="slepian-wolf",
    )
    tau = [-np.inf, 0.0, np.inf]
    leak = sr._leak_from_slepian_wolf(tau_arr=tau, quadrature_points=120)
    assert leak < 1.0


def test_leak_from_slepian_wolf(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    leak = sr_instance._leak_from_slepian_wolf(tau_arr=tau, quadrature_points=40)
    assert leak >= 0.0


def test_evaluate_key_rate(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    key_rate = sr_instance.evaluate_key_rate_in_bits_per_pulse(tau)
    assert np.isfinite(key_rate)

def test_eta_definition_matches_bits(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    metrics = sr_instance.evaluate_reconciliation_efficiency(tau, code_efficiency=0.95)
    numerator = metrics["bits_sent"] - metrics["bits_leaked"]
    assert metrics["eta"] * metrics["I_AB"] == pytest.approx(numerator, rel=1e-6)



def test_key_rate_components_align_with_eta(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    components = sr_instance.evaluate_key_rate_components(tau, code_efficiency=0.95)
    numerator = components["bits_sent"] - components["bits_leaked"]
    assert components["key_rate"] == pytest.approx(numerator - components["holevo"], rel=1e-6)



@pytest.mark.parametrize('m', range(1, 6))
def test_slepian_wolf_equiprobable_bins_independent_channel(m):
    sr = SR(
        m=m,
        modulation_variance=2.0,
        transmittance=0.0,
        excess_noise=0.01,
        code_efficiency=0.95,
        leak_strategy="slepian-wolf",
    )
    num_bins = 2 ** m
    tau = [-np.inf]
    sigma = np.sqrt(sr.modulation_variance)
    for i in range(1, num_bins):
        prob = i / num_bins
        edge = sr.px_rv.ppf(prob) / sigma
        tau.append(edge)
    tau.append(np.inf)

    quantisation_entropy = sr.evaluate_quantisation_entropy(tau)
    assert quantisation_entropy == pytest.approx(float(m), rel=1e-6)
    leak = sr._leak_from_slepian_wolf(tau_arr=tau, quadrature_points=200)
    assert leak == pytest.approx(quantisation_entropy, rel=2e-3)


def test_error_rate_eta_decreases_with_code_efficiency(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    eta_hi = sr_instance.evaluate_reconciliation_efficiency(tau, code_efficiency=0.95)["eta"]
    eta_lo = sr_instance.evaluate_reconciliation_efficiency(tau, code_efficiency=0.80)["eta"]
    assert eta_lo < eta_hi


def test_bits_leakage_matches_sum_of_slice_leaks(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    metrics = sr_instance.evaluate_reconciliation_efficiency(tau, code_efficiency=0.95)
    expected_leak = sum(slice_info["leak"] for slice_info in metrics["slices"])
    assert metrics["bits_leaked"] == pytest.approx(expected_leak, rel=1e-6)


def test_slice_leak_matches_eta_capacity_formula(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    metrics = sr_instance.evaluate_reconciliation_efficiency(tau, code_efficiency=0.95)
    for slice_info in metrics["slices"]:
        eta_c = slice_info["coding_efficiency"]
        capacity = slice_info["capacity"]
        expected = max(1.0 - eta_c * capacity, 0.0)
        assert slice_info["leak"] == pytest.approx(expected, rel=1e-6)
        assert expected == pytest.approx(1.0 - slice_info["code_rate"], rel=1e-6)


def test_leak_helper_matches_metrics(sr_instance):
    tau = [-np.inf, 0.0, np.inf]
    metrics = sr_instance.evaluate_reconciliation_efficiency(tau, code_efficiency=0.95)
    raw_errors = [slice_info["raw_error_rate"] for slice_info in metrics["slices"]]
    coding_efficiencies = [slice_info["coding_efficiency"] for slice_info in metrics["slices"]]
    helper = sr_instance._leak_from_error_rate(
        slice_error_rates=raw_errors, coding_efficiencies=coding_efficiencies
    )
    assert helper == pytest.approx(metrics["bits_leaked"], rel=1e-6)


def test_van_assche_m1_snr3_slice_error():
    sr = SR(
        m=1,
        modulation_variance=6.0,
        transmittance=1.0,
        excess_noise=0.0,
        code_efficiency=1.0,
    )
    assert sr.SNR == pytest.approx(3.0)
    tau = [-np.inf, 0.0, np.inf]
    e = sr._evaluate_slice_error_rates(tau)[0]
    assert e == pytest.approx(1.0 / 6.0, rel=2e-2, abs=2e-3)
    h_e = -e * math.log(e, 2) - (1 - e) * math.log(1 - e, 2)
    assert h_e == pytest.approx(0.65, rel=5e-3)


def test_van_assche_m4_snr3_profile():
    modulation_variance = 6.0
    transmittance = 1.0
    excess_noise = 0.0

    sr = SR(
        m=4,
        modulation_variance=modulation_variance,
        transmittance=transmittance,
        excess_noise=excess_noise,
        code_efficiency=1.0,
    )
    assert sr.SNR == pytest.approx(3.0, rel=1e-9)

    raw_tau = np.array([
        -np.inf,
        -2.347,
        -1.808,
        -1.411,
        -1.081,
        -0.768,
        -0.514,
        -0.254,
        0.0,
        0.254,
        0.514,
        0.768,
        1.081,
        1.411,
        1.808,
        2.347,
        np.inf,
    ])
    sigma = np.sqrt(modulation_variance)
    tau = np.array([
        value if not np.isfinite(value) else value / sigma for value in raw_tau
    ])

    errors = sr._evaluate_slice_error_rates(tau, bit_assignment="binary")
    expected_errors = [0.36348136, 0.32610279, 0.26102742, 0.15641925]
    for observed, expected in zip(errors, expected_errors):
        assert observed == pytest.approx(expected, rel=1e-6, abs=1e-6)

    metrics = sr.evaluate_reconciliation_efficiency(
        tau, code_efficiency=1.0, bit_assignment="binary"
    )
    assert metrics["quantisation_entropy"] == pytest.approx(3.7714647505, rel=5e-4)
    expected_leak = sum(slice_info["leak"] for slice_info in metrics["slices"])
    assert metrics["bits_leaked"] == pytest.approx(expected_leak, rel=1e-6)
    net_bits = metrics["bits_sent"] - metrics["bits_leaked"]
    assert net_bits == pytest.approx(0.4610678734, rel=5e-4)
