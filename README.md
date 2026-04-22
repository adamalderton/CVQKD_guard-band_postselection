# CVQKD Guard-Band Postselection

This repository contains a compact set of continuous-variable QKD reconciliation models, centered on the classic `GBSR`, `SR`, and `MD` implementations, plus a newer experimental quantized rewrite kept for research work.

The code is research-oriented rather than production-oriented. The goal is to make the main implementations, tests, and supporting exploratory material easy to scan quickly.

## Repo Map

- `guard_band_postselection/`: protocol implementations and shared base code
- `tests/`: tests for the classic protocol modules and the experimental rewrite
- `research/`: curated notebooks and dimension-reduction exploration

## Public API

The main public imports are:

```python
from guard_band_postselection import KeyEfficiencyBase, GBSR, SR, MD
```

These correspond to:

- [guard_band_postselection/key_efficiency_base.py](guard_band_postselection/key_efficiency_base.py)
- [guard_band_postselection/GBSR.py](guard_band_postselection/GBSR.py)
- [guard_band_postselection/SR.py](guard_band_postselection/SR.py)
- [guard_band_postselection/MD.py](guard_band_postselection/MD.py)

The newer research-only rewrite is also exposed as:

```python
from guard_band_postselection import QuantizedGBSR
```

and is implemented in [guard_band_postselection/quantized_gbsr.py](guard_band_postselection/quantized_gbsr.py).

## Install

Install the repository dependencies with:

```bash
pip install -r requirements.txt
```

## Minimal Usage

Example: instantiate the classic guard-band sliced reconciliation model and evaluate its reconciliation metrics.

```python
from guard_band_postselection import GBSR

gbsr = GBSR(
    m=1,
    modulation_variance=2.0,
    transmittance=0.5,
    excess_noise=0.01,
    code_efficiency=0.95,
)

tau = [-float("inf"), 0.0, float("inf")]
guards = [[0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
metrics = gbsr.evaluate_reconciliation_efficiency(tau, guards)
```

If you want the newer quantized model used in the research scripts, import `QuantizedGBSR` explicitly.

## Research Material

Curated research assets live under `research/`:

- `research/notebooks/`: retained exploratory notebooks
- `research/dimension_reduction/`: dimension-reduction scripts and notes

These files are kept to explain the current implementation and preserve a small amount of reproducibility context. They are not meant to be a complete archive of every exploratory step taken during development.

## Out of Scope

- full packaging and distribution metadata
- production hardening
- a complete historical archive of every legacy implementation path
- tracked generated figures and raw output tables
