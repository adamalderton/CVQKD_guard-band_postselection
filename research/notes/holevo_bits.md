# Holevo with Bit Register after Post-Selection (one-pass, bit-label agnostic)

These notes define and evaluate the post-selected Holevo information when reconciliation is performed **in one pass** over an $m$-bit quantizer of a single filtered quadrature (per-axis post-selection). We begin from the continuous-register construction to define the average post-selected Eve state $\rho_E^{\mathrm{PS}}$, then build the **bit-label–agnostic** Holevo $\chi_{\text{bits}}^{\mathrm{PS}}$ for a $2^m$-ary register without committing to Gray or natural labeling.

---

## 1. Setup and the continuous-register ensemble

- Bob heterodynes: write the real variables $Y=(y_1,y_2)^\top$ and the complex outcome $\beta=\tfrac{y_1+i y_2}{2}$.
- **Per-axis post-selection:** keep an outcome iff a 1D predicate $F(y_1)\in\{0,1\}$ holds (guard bands on $y_1$). Define the 1D pass probability
  $$
  p \equiv p_{\mathrm{pass}}=\int_{\mathbb{R}} F(y)\,p_Y(y)\,dy,
  $$
  where $p_Y$ is Bob’s per-quadrature heterodyne density (Gaussian). The other axis $y_2$ is **not** filtered and is integrated over $\mathbb{R}$.

- Under a collective Gaussian attack, Eve’s state conditioned on $\beta$ is a **displaced** copy of a fixed Gaussian seed $\rho_{E\mid 0}$ via the operator $D\!\big(\tfrac12\,\Gamma\,\beta\big)$, where $\Gamma$ is the usual linear MMSE map from Bob to Eve.

### Post-selected **average** Eve state (continuous-register construction)
$$
\boxed{
\rho_E^{\mathrm{PS}}
=\frac{1}{p}\!\int_{\mathbb{R}} F(y_1)\,p_Y(y_1)\!
\left[\int_{\mathbb{R}} p_Y(y_2)\,
D\!\Big(\tfrac12\,\Gamma\,\beta\Big)\,\rho_{E\mid 0}\,D^\dagger\!\Big(\tfrac12\,\Gamma\,\beta\Big)\,dy_2\right]dy_1,\quad
\beta=\frac{y_1+i y_2}{2}.
}
$$

### Continuous-register post-selected Holevo
The cq state is $\rho^{\mathrm{PS}}_{E\beta}=\int p(\beta\mid\mathrm{PS})\,|\beta\rangle\langle\beta|\otimes \rho_{E\mid\beta}\,d^2\beta$. Displacement invariance of conditional entropies under Gaussian attacks gives $S(\rho_{E\mid\beta})=S(\rho_{E\mid 0})$, hence
$$
\boxed{
\chi_\beta^{\mathrm{PS}}=\chi(E;\beta\mid\mathrm{PS})
= S\!\big(\rho_E^{\mathrm{PS}}\big) - S(E\mid 0).
}
$$

This object is **bit-assignment agnostic** and depends only on the pass set $F$ and the channel parameters.

---

## 2. Quantization in **one pass**: the $2^m$-ary register $J$ and $\chi_{\text{bits}}^{\mathrm{PS}}$

Partition the filtered axis into $2^m$ disjoint intervals $\{I_j\}_{j=0}^{2^m-1}$. After guard bands, keep $I_j^{\mathrm{pass}}\subseteq I_j$. Define the **symbol register**
$$
J\in\{0,\dots,2^m\!-\!1\},\qquad J=j \iff y_1\in I_j^{\mathrm{pass}}.
$$

- **Weights on accepted intervals**
  $$
  p(j\mid \mathrm{PS})
  =\frac{\int_{I_j^{\mathrm{pass}}} p_Y(y_1)\,dy_1}{p},
  \qquad \sum_j p(j\mid\mathrm{PS})=1.
  $$

- **Interval-conditioned Eve states** (other quadrature integrated out)
  $$
  \boxed{
  \rho^{\mathrm{PS}}_{E\mid j}
  =\frac{\int_{I_j^{\mathrm{pass}}}\! p_Y(y_1)\,\left[\int_{\mathbb{R}} p_Y(y_2)\,D(\tfrac12\Gamma\beta)\,\rho_{E\mid 0}\,D^\dagger(\tfrac12\Gamma\beta)\,dy_2\right]dy_1}
  {\int_{I_j^{\mathrm{pass}}}p_Y(y_1)\,dy_1}.}
  $$

- **Post-selected cq state with a $2^m$-ary classical register**
  $$
  \boxed{
  \rho^{\mathrm{PS}}_{EJ}
  =\sum_{j=0}^{2^m-1} p(j\mid\mathrm{PS})\,|j\rangle\!\langle j|\otimes \rho^{\mathrm{PS}}_{E\mid j},
  \qquad
  \rho_E^{\mathrm{PS}}=\sum_j p(j\mid\mathrm{PS})\,\rho^{\mathrm{PS}}_{E\mid j}.}
  $$

### One-pass (all bits at once) Holevo — **bit-label agnostic**
If the $m$ bits are reconciled in **one pass**, the relevant Eve term is the Holevo against the **entire** $m$-bit vector. Let $f_B$ be any bijection from $\{0,\dots,2^m-1\}$ to $\{0,1\}^m$ (Gray, natural, etc.) and define $\mathbf B=f_B(J)$. Since $f_B$ is invertible,
$$
\chi(E;\mathbf B\mid\mathrm{PS})=\chi(E;J\mid\mathrm{PS}).
$$
Therefore the **bit-register Holevo** is
$$
\boxed{
\chi_{\text{bits}}^{\mathrm{PS}}=\chi(E;J\mid\mathrm{PS})
= S\!\big(\rho_E^{\mathrm{PS}}\big)\;-\!\sum_{j=0}^{2^m-1} p(j\mid\mathrm{PS})\,S\!\big(\rho^{\mathrm{PS}}_{E\mid j}\big).
}
$$

**Bit-label agnosticism.** The formula uses only $\{p(j\mid\mathrm{PS}),\,\rho^{\mathrm{PS}}_{E\mid j}\}$, which are **independent of the labeling** $f_B$. Relabeling intervals simply permutes the set $\{(p(j\mid\mathrm{PS}),\rho^{\mathrm{PS}}_{E\mid j})\}$ and leaves the sum invariant. Hence $\chi_{\text{bits}}^{\mathrm{PS}}$ does **not** depend on Gray vs. natural coding when reconciliation is one-pass.

*(Only slice-by-slice decoding, or revealing per-slice side information, would introduce a dependence on labeling through conditional Holevo terms $\chi(E;B_k\mid B_{<k})$. We ignore that here.)*

---

## 3. How to evaluate the entropies — exact vs Gaussian surrogate

Each component of the mixtures above is a **displaced Gaussian** $D(\cdot)\rho_{E\mid 0}D^\dagger(\cdot)$. The **mixtures** $\rho_E^{\mathrm{PS}}$ and $\rho^{\mathrm{PS}}_{E\mid j}$ are generally **non-Gaussian** due to the guard-band geometry.

- **Exact (truncated-Fock) route.** Discretize each accepted set, apply displacements in a truncated Fock basis to $\rho_{E\mid 0}$, average to form $\rho_E^{\mathrm{PS}}$ and $\rho^{\mathrm{PS}}_{E\mid j}$, diagonalize to get $S(\cdot)$, and plug the results into the boxed formula for $\chi_{\text{bits}}^{\mathrm{PS}}$.

- **Gaussian surrogate (fast, conservative).** Replace each mixture by the **Gaussian with the same covariance**. By Gaussian extremality,
  $$
  S(\text{true mixture})\ \le\ S_{\mathrm G}(\text{covariance match}).
  $$
  In a phase-insensitive, per-axis setting one can use the scalar surrogate
  $$
  \nu_{\mathrm{PS}}\approx v_{\mathrm{cond}}+\kappa^2\,\frac{\mathrm{Var}(y\mid\mathrm{PS})+\mathrm{Var}(y)}{2},
  \qquad
  \nu_{j}\approx v_{\mathrm{cond}}+\kappa^2\,\frac{\mathrm{Var}(y\mid j)+\mathrm{Var}(y)}{2},
  $$
  and then
  $$
  S(\rho_E^{\mathrm{PS}})\approx g(\nu_{\mathrm{PS}}),
  \qquad
  S(\rho^{\mathrm{PS}}_{E\mid j})\approx g(\nu_j),
  $$
  giving the conservative evaluation
  $$
  \boxed{
  \chi_{\text{bits}}^{\mathrm{PS}}\ \approx\ g(\nu_{\mathrm{PS}})\;-\!\sum_{j=0}^{2^m-1} p(j\mid\mathrm{PS})\,g(\nu_j).
  }
  $$
  Here $v_{\mathrm{cond}}=a-\dfrac{c^2}{b+1}$ and $\kappa=\dfrac{c}{b+1}$ for Laudenbach’s heterodyne convention; $g(\nu)$ is the bosonic entropy function.

---

## 4. Relationship to the continuous-register Holevo (sanity check)

The map “quantize the accepted axis” is a classical channel $\Lambda:\beta\mapsto J$. For the cq state $\rho^{\mathrm{PS}}_{E\beta}$,
$$
\rho^{\mathrm{PS}}_{EJ}=(\mathrm{id}_E\otimes \Lambda)\big(\rho^{\mathrm{PS}}_{E\beta}\big),
$$
so data processing (monotonicity of relative entropy) yields
$$
\boxed{
\chi_{\text{bits}}^{\mathrm{PS}}=\chi(E;J\mid\mathrm{PS})\ \le\ \chi(E;\beta\mid\mathrm{PS})=\chi_\beta^{\mathrm{PS}}.
}
$$
Thus the one-pass, bit-label–agnostic $\chi_{\text{bits}}^{\mathrm{PS}}$ is always **tighter** than the continuous-register quantity and is the correct object to use when reconciliation operates on the full $m$-bit symbol in a single round.
