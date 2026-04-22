# CVQKD Guard-Band Postselection

This repository contains a compact set of continuous-variable QKD reconciliation models, centered on the legacy `GBSR`, `SR`, and `MD` implementations plus a newer experimental quantized rewrite retained for research work.

The code is research-oriented rather than production-oriented. The aim is to make the main implementation, assumptions, and supporting exploration easy to scan quickly.

## Repo Map

- `guard_band_postselection/`: protocol implementations and shared base code
- `tests/`: tests for the restored protocol modules and the experimental rewrite
- `research/`: curated notebooks, technical notes, and dimension-reduction exploration

## Core Module

The main public imports are:

```python
from guard_band_postselection import KeyEfficiencyBase, GBSR, SR, MD
```

The restored protocol modules live in:

- [guard_band_postselection/key_efficiency_base.py](guard_band_postselection/key_efficiency_base.py)
- [guard_band_postselection/GBSR.py](guard_band_postselection/GBSR.py)
- [guard_band_postselection/SR.py](guard_band_postselection/SR.py)
- [guard_band_postselection/MD.py](guard_band_postselection/MD.py)

The newer research-only rewrite remains available as:

```python
from guard_band_postselection import QuantizedGBSR
```

and is implemented in [guard_band_postselection/quantized_gbsr.py](guard_band_postselection/quantized_gbsr.py).

## Install

Create an environment and install the lightweight runtime dependencies:

```bash
pip install -r requirements.txt
```

## Minimal Usage

```python
from guard_band_postselection import GBSR

gbsr = GBSR(
    m=1,
    modulation_variance=1.0,
    transmittance=0.1,
    excess_noise=0.001,
)

tau = gbsr.build_equiprobable_tau()
guards = gbsr.generate_g_arr_from_p_pass(0.9, tau)
metrics = gbsr.evaluate_quantised_maximum_key_efficiency(tau, guards)
```

## Research Material

Curated research assets live under `research/`:

- `research/notebooks/`: retained exploratory notebooks
- `research/dimension_reduction/`: dimension-reduction scripts and notes
- `research/notes/`: supporting derivations and technical notes

These files are kept to explain the current implementation and preserve a small amount of reproducibility context. They are not meant to be a complete archive of every exploratory step taken during development.

Research code under `research/` that depends on the newer rewrite should import `QuantizedGBSR` explicitly rather than the default `GBSR`.

## Out of Scope

- full packaging and distribution metadata
- production hardening
- a complete historical archive of every legacy implementation path
- tracked generated figures and raw output tables
