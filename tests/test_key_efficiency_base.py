import math

import numpy as np
import pytest

from guard_band_postselection.key_efficiency_base import (
    KeyEfficiencyBase,
    _rect_prob,
)


class DummyKeyEfficiency(KeyEfficiencyBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("leak_strategy", "error-rate")
        super().__init__(*args, **kwargs)

    def evaluate_key_rate_in_bits_per_pulse(self, *args, **kwargs) -> float:  # pragma: no cover
        return 0.0

    def _leak_from_error_rate(self, *, error_rate: float, **_kwargs) -> float:
        return error_rate

    def _leak_from_slepian_wolf(self, **_kwargs) -> float:
        return 42.0

    def _leak_from_capacity(self, **_kwargs) -> float:
        return 0.5


@pytest.fixture()
def base_instance():
    # Use zero transmittance so Alice and Bob variables decouple, simplifying expectations.
    return DummyKeyEfficiency(
        modulation_variance=2.0,
        transmittance=0.0,
        excess_noise=0.1,
        Delta_QCT=0.25,
    )


def test_leak_strategy_dispatch(base_instance):
    assert base_instance._compute_reconciliation_leak(error_rate=0.3) == pytest.approx(0.3)
    base_instance.set_leak_strategy("slepian-wolf")
    assert base_instance._compute_reconciliation_leak() == pytest.approx(42.0)
    base_instance.set_leak_strategy("capacity")
    assert base_instance._compute_reconciliation_leak() == pytest.approx(0.5)
    base_instance.set_leak_strategy("error-rate")


def test_build_joint_covariance(base_instance):
    cov = base_instance._build_joint_covariance()
    expected = np.array([[2.0, 0.0], [0.0, base_instance.bob_variance]])
    assert np.allclose(cov, expected)


def test_evaluate_quantisation_entropy(base_instance):
    entropy = base_instance.evaluate_quantisation_entropy([-np.inf, 0.0, np.inf])
    assert pytest.approx(entropy, rel=1e-6) == 1.0



def test_evaluate_devetak_winter_defaults(base_instance):
    beta = 0.8
    expected = beta * base_instance.I_AB - base_instance._holevo_with_qct()
    assert base_instance.evaluate_devetak_winter(beta) == pytest.approx(expected)

def test_evaluate_devetak_winter_override(base_instance):
    value = base_instance.evaluate_devetak_winter(
        0.5, mutual_information=1.2, holevo=0.3
    )
    assert value == pytest.approx(0.5 * 1.2 - 0.3)

def test_evaluate_holevo_information_matches_g_formula(base_instance):
    holevo = base_instance._evaluate_holevo_information()
    a, b, c = base_instance.a, base_instance.b, base_instance.c
    sqrt_value = (a + b) ** 2 - (4.0 * c ** 2)
    nu0 = 0.5 * (np.sqrt(sqrt_value) + (b - a))
    nu1 = 0.5 * (np.sqrt(sqrt_value) - (b - a))
    nu2 = a - (c ** 2) / (b + 1.0)
    manual = sum(base_instance._g(val) for val in (nu0, nu1)) - base_instance._g(nu2)
    assert pytest.approx(holevo, rel=1e-12) == manual


def test_holevo_with_qct_adds_offset(base_instance):
    plain = base_instance._evaluate_holevo_information()
    with_qct = base_instance._holevo_with_qct()
    assert pytest.approx(with_qct, rel=1e-12) == plain + base_instance.Delta_QCT


def test_g_function_behaviour(base_instance):
    assert base_instance._g(0.5) == 0.0
    value = base_instance._g(2.0)
    expected = ((2.0 + 1.0) / 2.0) * math.log2((2.0 + 1.0) / 2.0) - ((2.0 - 1.0) / 2.0) * math.log2(
        (2.0 - 1.0) / 2.0
    )
    assert pytest.approx(value, rel=1e-12) == expected


def test_binary_entropy_properties(base_instance):
    assert pytest.approx(base_instance._binary_entropy(0.5), rel=1e-12) == 1.0
    with pytest.raises(ValueError):
        base_instance._binary_entropy(-0.1)


def test_integrate_1d_gaussian_pdf(base_instance):
    rv = base_instance.px_rv
    prob = base_instance._integrate_1D_gaussian_pdf(rv, [0.0, np.inf])
    assert pytest.approx(prob, rel=1e-6) == 0.5


def test_integrate_2d_gaussian_pdf(base_instance):
    prob = base_instance._integrate_2D_gaussian_pdf(
        base_instance.joint_rv,
        [0.0, np.inf],
        [0.0, np.inf],
    )
    assert pytest.approx(prob, rel=1e-6) == 0.25


def test_generate_gray_bit_assignment(base_instance):
    assert base_instance._generate_gray_bit_assignment(2) == ["00", "01", "11", "10"]


def test_generate_binary_bit_assignment(base_instance):
    assert base_instance._generate_binary_bit_assignment(3) == ["000", "001", "010", "011", "100", "101", "110", "111"]


def test_hamming_distance(base_instance):
    assert base_instance._hamming_distance("0101", "1100") == 2


def test_rect_prob_matches_independent_quadrant_probability():
    mean_x = 0.0
    mean_y = 0.0
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    prob = _rect_prob(mean_x, mean_y, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1], 0.0, np.inf, 0.0, np.inf)
    assert pytest.approx(prob, rel=1e-6) == 0.25




@pytest.mark.parametrize('m', range(1, 6))
def test_quantisation_entropy_equiprobable_bins(base_instance, m):
    num_bins = 2 ** m
    tau = [-np.inf]
    for i in range(1, num_bins):
        prob = i / num_bins
        edge = base_instance.px_rv.ppf(prob) / np.sqrt(base_instance.modulation_variance)
        tau.append(edge)
    tau.append(np.inf)

    entropy = base_instance.evaluate_quantisation_entropy(tau)
    assert entropy == pytest.approx(float(m), rel=1e-6)
