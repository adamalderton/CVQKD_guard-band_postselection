# Practical Notes on Optimising the Sliced Reconciliation (SR) Key Rate

> Focus: General $m$-slice formulation with practical emphasis on $m=2$.
>
> Code reference: `guard_band_postselection/SR.py` and shared machinery in `key_efficiency_base.py`.

---
## 1. Context and Goal
Continuous-Variable Quantum Key Distribution (CV-QKD) with *sliced reconciliation* (Van Assche et al., 2004) converts correlated Gaussian variables $(X, Y)$ held by Alice and Bob into $m$ binary slices per raw symbol. Each slice is reconciled using (practical) forward error-correcting codes (e.g. LDPC). The **design (optimisation) problem** is to choose the quantisation (interval) boundaries that map $X$ into $2^m$ disjoint intervals so that the *practical* secret key rate per pulse is maximised under finite coding efficiency.

We assume (as in the code) a Gaussian modulation with variance $V_A = \sigma_X^2 = V_\text{mod}$ and Bob's measurement variance
$$V_B = \sigma_Y^2 = \tfrac{1}{2} T V_A + 1 + \tfrac{1}{2} \xi,$$
with transmittance $T$, excess noise $\xi$ (referred to the channel input), and shot-noise units set so that one unit of vacuum noise is 1 (heterodyne adds an extra unit overall—handled in code). The signal-to-noise ratio and mutual information without post-selection are
$$\mathrm{SNR} = \frac{ \tfrac{1}{2} T V_A }{1 + \tfrac{1}{2} \xi}, \qquad I_{AB} = \log_2(1 + \mathrm{SNR}).$$

We focus on *sliced* (not guard-band) reconciliation. Guard bands introduce erasures / post-selection (handled in `GBSR.py`); SR keeps every symbol.

---
## 2. Quantisation and Variables
Let $m \ge 1$ be the number of slices. Define $K = 2^m$ intervals on Alice's variable $X$ via edges
$$-\infty = \tau_0 < \tau_1 < \dots < \tau_{K-1} < \tau_K = +\infty.$$
(Practically we work with *normalised* edges $t_i$ in code (the `tau_arr` passed into `SR.evaluate_*` is already scaled by $\sqrt{V_A}$ when used), but conceptually we treat $\tau_i$ directly in $X$'s physical units.)

Each $x$ is mapped to an interval index $s \in \{0, 1, \dots, K-1\}$ that is then *labelled* by an $m$-bit string. The implementation supports binary or Gray ordering:
- Gray labelling minimises Hamming distance between neighbouring intervals, thereby reducing high-weight error events for small cross-over probabilities.
- Binary labelling is simpler but can inflate slice error probabilities at higher significance levels.

### 2.1 Slice Decoding Order
In `SR.py`, slices are decoded **least significant to most significant** (LSB $\to$ MSB). This order matters because once lower slices are corrected, they effectively refine Bob's a posteriori knowledge for higher slices (the classical multi-stage decoding picture). The error-rate integration in `_evaluate_slice_error_rates` respects this ordering by iterating bit indices in reversed order.

---
## 3. Probabilistic Structure
The joint distribution $(X, Y)$ is zero-mean Gaussian with covariance
$$\Sigma = \begin{pmatrix} V_A & C_{XY} \\ C_{XY} & V_B \end{pmatrix}, \qquad C_{XY} = \sqrt{\tfrac{T}{2}}\, V_A.$$
Conditionally,
$$X \mid Y=y \sim \mathcal{N}\Big( \mu(y), \sigma_{X|Y}^2 \Big),\quad \mu(y) = \frac{C_{XY}}{V_B} y,\quad \sigma_{X|Y}^2 = V_A - \frac{C_{XY}^2}{V_B}. $$
This conditional distribution underpins the slice error probability integrals.

The probability that $X$ lies in interval $i$ (marginal) is
$$p_i = \Pr( \tau_i < X \le \tau_{i+1}) = \Phi\Big( \frac{\tau_{i+1}}{\sigma_X} \Big) - \Phi\Big( \frac{\tau_i}{\sigma_X} \Big),$$
where $\Phi$ is the standard normal CDF.

The **quantisation entropy** is
$$H_Q = H(S) = - \sum_{i=0}^{K-1} p_i \log_2 p_i,$$
implemented in `KeyEfficiencyBase.evaluate_quantisation_entropy`.

---
## 4. Slice Error Rates
For slice (bit) $k$ ($k=0$ least significant), define the induced binary variable $B_k$ from the $k^{th}$ bit of the interval label. The *raw* per-slice error rate (prior to folding to $[0, 0.5]$) is
$$e_k = \Pr(B_k^{(\text{Alice})} \neq \hat B_k^{(\text{Bob})}),$$
obtained by integrating (over $Y$) the minimum of the two conditional likelihoods of the bit given $Y$, aggregated over prefixes of previously decoded slices.

Implementation outline (see `_evaluate_slice_error_rates`):
1. For each decoding stage (from LSB upward) collect all intervals sharing the same already-decoded prefix.
2. For each such *prefix group*, partition intervals according to current bit value (0 vs 1) and integrate the conditional Gaussian mass contribution on $X$ between each interval's edges for a grid of $Y$ samples (Gauss–Hermite quadrature over $Y$'s marginal).
3. For each $Y$ sample accumulate $g_0(y)$ and $g_1(y)$ — unnormalised densities of the bit being 0 or 1.
4. The instantaneous bit error contribution at $y$ is $\min(g_0(y), g_1(y))$; integrate over $y$ with weights to get $e_k$.

### Folding and Capacity
After computing raw $e_k$, the *effective* BSC error is
$$\tilde e_k = \min(e_k, 1 - e_k).$$
The channel capacity of slice $k$ is
$$C_k = 1 - h_2(\tilde e_k),$$
where $h_2$ is the binary entropy function.

If the coding efficiency of the practical FEC for slice $k$ is $\eta_c(e_k) \in (0,1]$, the realised code rate is
$$R_k = \eta_c(e_k) \; C_k.$$
The leakage (in bits) consumed by revealing syndrome / parity information for slice $k$ is
$$\ell_k = 1 - R_k,$$
clipped to $[0,1]$ in code for numerical robustness.

---
## 5. Objective: Practical Key Rate
All $m$ slices are always kept (no post-selection). The total bits **sent** per raw symbol are the quantisation index bits produced: $H_Q$ (not always equal to $m$ if intervals are imbalanced). Total **leaked** bits are $L = \sum_{k=1}^m \ell_k$.

The (per-symbol) private reconciled bits prior to privacy amplification are
$$B_{\text{priv}} = H_Q - L.$$
The final key rate subtracts Eve's Holevo information (Gaussian collective attack model + optional QCT penalty $\Delta_{\mathrm{QCT}}$):
$$K = B_{\text{priv}} - (\chi_{EB} + \Delta_{\mathrm{QCT}}).$$
In code (`SR.evaluate_reconciliation_efficiency`) this is
```
key_rate = (bits_sent - bits_leaked) - holevo_with_qct.
```
Additionally, a reconciliation efficiency metric
$$\eta = \frac{H_Q - L}{I_{AB}}$$
is reported for comparison with the ideal scaling argument $K_{\text{ideal}} = I_{AB} - \chi$.

### Optimisation Problem (General m)
We seek interval edges $\boldsymbol{\tau} = (\tau_1, \dots, \tau_{K-1})$ that maximise
$$\max_{\tau_1 < \dots < \tau_{K-1}} \; F(\boldsymbol{\tau}) = \Big[ H_Q(\boldsymbol{\tau}) - \sum_{k=1}^m \ell_k(\boldsymbol{\tau}) \Big] - (\chi_{EB} + \Delta_{\mathrm{QCT}}).$$
The Holevo term here is independent of the quantiser (for plain SR without post-selection), so the *effective* optimisation is
$$\max_{\boldsymbol{\tau}} \; H_Q(\boldsymbol{\tau}) - \sum_{k=1}^m \ell_k(\boldsymbol{\tau}).$$
Equivalently minimise (interpreting leak as overhead)
$$\min_{\boldsymbol{\tau}} \; \sum_{k=1}^m \ell_k(\boldsymbol{\tau}) - H_Q(\boldsymbol{\tau}).$$

### Degrees of Freedom and Reparameterisation
Because tails extend to $\pm\infty$, only *finite* internal edges are optimised. For numerical stability:
- Parameterise unconstrained vector $z \in \mathbb{R}^{K-1}$ and set
  $$\tau_i = f(z)_i = \mu_X + \sigma_X \; \mathrm{erf}^{-1}\big( 2 \Phi_i - 1 \big),$$
  where $(\Phi_1, \dots, \Phi_{K-1})$ is the cumulative sum of softmax-transformed positive masses ensuring ordering.
- Simpler (used implicitly in experiments): optimise directly over an ordered list by constraining differences $d_i = \tau_i - \tau_{i-1} > 0$ with $d_i = \exp(w_i)$.

---
## 6. Structure and Challenges of the Objective
### 6.1 Competing Effects
- Pushing edges outward balances interval probabilities (increasing $H_Q$ toward $m$) but also changes conditional overlap between bit partitions, possibly increasing certain slice error probabilities and hence leakage.
- Highly imbalanced intervals reduce $H_Q$ drastically, usually degrading the net key despite potentially lower error on more significant slices.

### 6.2 Non-Convexity
The mapping $\boldsymbol{\tau} \mapsto (e_1, \dots, e_m)$ involves nested integrals of minima of *mixtures* of truncated Gaussians. There is no convexity guarantee. Practically:
- Multiple local maxima appear for larger $m$ or low SNR.
- For $m=2$, the landscape is smoother; heuristic searches (grid initialisation + local refinement) usually converge reliably.

### 6.3 Scale and Conditioning
Edges may naturally lie within a few standard deviations ($\pm (2\text{--}5)\sigma_X$). Using raw physical units can yield badly scaled gradients if autodiff were applied. Normalising by $\sigma_X$ (as in code: `tau_arr` is a *normalised* array passed in and then multiplied by $\sqrt{V_A}`) improves conditioning.

### 6.4 Gradient Availability
The present implementation is *not* differentiable end-to-end via automatic differentiation because:
- It uses SciPy's Gauss–Hermite nodes *outside* an autodiff framework.
- Takes `min(g0, g1)` which is non-smooth at equality points.
- Employs explicit Python loops over prefixes.

To obtain gradients one could:
1. Replace integration with differentiable quadrature under JAX / PyTorch and vectorise prefix handling.
2. Approximate $\min(a,b)$ with smooth soft-min: $\operatorname{softmin}_\beta(a,b) = -\beta^{-1} \log(e^{-\beta a} + e^{-\beta b})$ for large $\beta$.
3. Use reparameterisation for edges to enforce ordering automatically.

Given the complexity, **derivative-free** optimisers (CMA-ES, Nelder–Mead, pattern search) or **hybrid** (coarse grid then BFGS on a smoothed surrogate) are pragmatic.

### 6.5 Numerical Integration Nuances
- Gauss–Hermite order (240 in code) is high: improves accuracy for tail events but increases cost. Reducing order during early search accelerates exploration; refine at the end.
- Catastrophic cancellation is mild here because we integrate CDF differences `ndtr(upper) - ndtr(lower)`; still, when intervals are very narrow relative to $\sigma_{X|Y}$, these differences can approach floating underflow. Clip with a floor (code uses positivity + accumulation in float64 implicitly via NumPy).
- When $e_k$ is extremely small ($<10^{-8}$) capacity saturates to 1; coding efficiency modelling may not be valid at that extreme—impose a lower bound on error so that $\eta_c(e)$ interpolation stays meaningful.

### 6.6 Error Folding and Plateau Effects
The folding $e \mapsto \tilde e = \min(e,1-e)$ creates a flat region of the objective whenever a slice crosses $0.5$. During optimisation if an interval configuration yields a slice with $e_k > 0.5$, local perturbations that keep it above 0.5 do not change capacity or leakage; the optimiser might stall. Mitigation:
- Penalise configurations with any $e_k > 0.5 - \epsilon$.
- Re-initialise from a balanced quantiser if encountered.

### 6.7 Correlation of Slice Errors
Slices are *not independent*: altering one boundary changes the mapping for all slice bits (due to relabelling structure). The sequential decoding order exacerbates coupling—optimising edges focusing only on the highest significance bit often degrades lower slices. A balanced approach considers the **sum leakage** directly.

---
## 7. Practical Optimisation Workflow (m=2 Focus)
For $m=2$ we have 4 intervals and two internal edges $\tau_1, \tau_2, \tau_3$ with $\tau_0=-\infty, \tau_4=+\infty$; by symmetry of $X$ one can restrict to an *anti-symmetric* pattern when optimality suggests it:
$$\tau_1 = -a,\; \tau_2=0,\; \tau_3= a.$$
This reduces the search to a single positive parameter $a > 0$ (empirically near $\approx 0.8\sigma_X$–$1.4\sigma_X$ depending on SNR). Steps:
1. Coarse grid over $a \in [0.3, 2.5]$ (normalised units). Compute objective.
2. Fit a smooth 1D surrogate (cubic spline) to objective vs $a$; maximise spline.
3. (Optional) Unlock central edge 0 and re-optimise full trio $(\tau_1, \tau_2, \tau_3)$ starting from symmetric solution using a local search.
4. Validate with higher Gauss–Hermite order.

For larger $m$, symmetry ansatz generalises: enforce mirror symmetry about 0 and monotonic spacing (denser near centre). One popular heuristic is to initialise internal cumulative probabilities as evenly spaced: $P(X \le \tau_i)= i/K$ (equiprobable quantiser), then refine.

---
## 8. Coding Efficiency Modelling
Coding efficiency $\eta_c(e)$ captures performance loss vs Shannon limit. In the current SR implementation a *per-slice* model can be:
- Constant (e.g. 0.95) — simple upper bound scenario.
- Function of slice error (mirroring LDPC operating curve). See `code_efficiency.py` for a wrapper around an analytic LDPC throughput model (`LDPCSimpleModel`).

Because slice error rates differ significantly (LSB typically worse), a per-slice adaptive $\eta_c(e_k)$ naturally allocates more leakage to noisy slices. The derivative of $R_k$ wrt $e_k$ is
$$\frac{\partial R_k}{\partial e_k} = \eta_c'(e_k) C_k + \eta_c(e_k) \frac{d C_k}{d e_k}, \qquad \frac{d C_k}{d e_k} = - h_2'(e_k^*) \cdot \frac{d e_k^*}{d e_k},$$
where $e_k^* = \tilde e_k$ and
$$h_2'(x) = \log_2\Big(\frac{1-x}{x}\Big).$$
When $e_k < 0.5$, $d e_k^* / d e_k = 1$; otherwise $=-1$. This derivative is useful if future gradient-based optimisation is pursued.

---
## 9. Suggested Enhancements for Gradient-Based Methods
1. **Vectorisation**: Express prefix grouping as a tensor contraction over interval-bit masks.
2. **Smooth Min Approximation**: Replace $\min(g_0,g_1)$ by soft-min with annealed $\beta$.
3. **Autodiff Backend**: Port to JAX; use `jax.scipy.stats.norm.cdf` for conditional integrals.
4. **Log-Space Stabilisation**: For extremely tight intervals compute $\log \Phi$ differences via `log1p` expansions.
5. **Adaptive Quadrature**: Start with 60 nodes; double near convergence.

---
## 10. Numerical Pitfalls and Mitigations
| Pitfall | Symptom | Mitigation |
|---------|---------|------------|
| Narrow interval near mean | $p_i \approx 0$ causing NaNs in $p_i \log p_i$ | Clip probabilities, skip zero terms (current code already filters zeros). |
| Extremely small $e_k$ | Capacity numerically 1.000.. causing flat objective | Enforce floor $e_k \ge 10^{-9}$ before capacity; or add small penalty to discourage over-refinement. |
| $e_k > 0.5$ region | Plateau due to folding | Penalise $e_k > 0.49$. |
| High Gauss–Hermite order | Slow iterations | Use progressive refinement / cache nodes. |
| Prefix explosion (large m) | O($2^m$) grouping cost | Precompute bit masks; prune negligible prefix probability groups. |
| Loss of ordering | Optimiser produces $\tau_{i+1} \le \tau_i$ | Use exponential increments or projection sorting after each step. |

---
## 11. Validation Metrics
After obtaining candidate edges:
1. Report $(H_Q, e_1, \dots, e_m, R_1, \dots, R_m, K)$.
2. Compare $H_Q$ to $m$; large gap indicates imbalanced quantiser.
3. Check all $e_k < 0.5$ (strict).
4. Sensitivity: perturb each $\tau_i$ by $\pm 1\%$ and ensure objective decreases (local maximum test).
5. Robustness: vary $\eta_c$ model parameters; optimal structure should be stable under mild perturbations.

---
## 12. Example (Hypothetical m=2 Flow)
Suppose $T=0.1, V_A=4.0, \xi=0.01$, $m=2$, constant $\eta_c=0.95$.
1. Equiprobable quantiser solution for 4 intervals: $p_i = 0.25$ gives edges at quantiles $(-\infty, q_{0.25}, q_{0.5}, q_{0.75}, +\infty)$ with $q_{0.5}=0$.
2. Evaluate slice errors; expect $e_0 > e_1$ (LSB noisier).
3. Small adjustment: widen central intervals slightly to reduce LSB error, trading a small decrease in $H_Q$ for larger leakage reduction (net gain in $K$).
4. Iterate until gain < tolerance.

---
## 13. Relationship to Guard-Band SR (GBSR)
In `GBSR.py`, additional *guard bands* remove regions of $Y$ near decision boundaries, introducing a pass probability $p_{\text{pass}} < 1$ that scales both useful bits and leakage. Optimisation then couples interval edges and guard band widths $(g_i^{\text{left}}, g_i^{\text{right}})$. The Holevo term may change if post-selection alters effective covariance ("optimistic" strategy). Plain SR is thus a special case with all guard widths zero and $p_{\text{pass}}=1$.

---
## 14. Summary Checklist for Practitioners
- Start from equiprobable quantiser; exploit symmetry when plausible.
- Use Gray labelling unless a coding scheme specifically prefers another mapping.
- Limit Gauss–Hermite nodes early; refine near convergence.
- Monitor each slice error; enforce $e_k < 0.5$ strictly.
- Use derivative-free global (CMA-ES) + local polishing.
- Consider reparameterisation to maintain ordering automatically.
- Validate stability under slight noise / parameter perturbations.

---
## 15. Possible Future Extensions
- Joint optimisation of $m$ (model selection) trading complexity vs leakage.
- Incorporating finite-size effects (statistical fluctuations in $e_k$ estimates).
- Bayesian update of interval edges based on observed empirical error rates mid-protocol.
- Adaptive per-slice code length / rate scheduling conditioned on live error estimates.
- Full autodiff pipeline enabling second-order methods (L-BFGS, quasi-Newton with Hessian-vector products).

---
## 16. References
- Van Assche, D. et al. (2004). *Quantum key distribution with continuous variables using coherent states and homodyne detection* (sliced reconciliation concept).
- Jouguet, P. et al. (2011, 2014). Practical CV-QKD implementations, coding efficiency considerations.
- Leverrier, A. (2008). *Reconciliation in CV-QKD*.
- Laudenbach, F. et al. (2018). Practical modelling of heterodyne / dual-homodyne detection.

---
*End of notes.*
