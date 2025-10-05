# Reconciliation Efficiency Notes (BSC one‑way LDPC model)

**Premise.** In what follows we compute reconciliation efficiency using the ratio
$$
\beta = \frac{\mathbf{bits\_sent} - \mathbf{bits\_leaked}}{I_{AB}},
$$
where **bits_leaked** is determined *practically*: we assume one‑way LDPC codes per slice, model each slice channel as a binary symmetric channel (BSC) with crossover probability $e$, and therefore count leakage using the binary entropy $h_2(e)$. No soft information, no interactive passes.

Throughout,
$$
h_2(e) = -e\log_2 e - (1-e)\log_2(1-e).
$$
For a code of capacity‑relative efficiency $\eta_c\in(0,1]$ on a BSC, the code rate is
$$
R(e,\eta_c)=\eta_c\big(1-h_2(e)\big),
$$
and the leak per binary symbol (syndrome length fraction) is
$$
\boxed{\text{LEAK}_{\rm EC}(e,\eta_c)=1-\eta_c\big(1-h_2(e)\big)}
= h_2(e)+(1-\eta_c)\big(1-h_2(e)\big).
$$
This reduces to $h_2(e)$ in the ideal $\eta_c\to 1$ limit.

We define the code-efficiency as
$$
\eta_c(e,S_{\min}) \equiv \frac{R^\star(e,S_{\min})}{C_{\text{BSC}}(e)},\qquad C_{\text{BSC}}(e)=1-h_2(e),
$$
where $R^\star(e,S_{\min})$ is the **highest practical code rate** that still meets the minimum decoder speed $S_{\min}$ at BER $e$. Thus $\eta_c\in[0,1]$ is your gap-to-capacity: $\eta_c=1$ is capacity-achieving; smaller $\eta_c$ means more leakage $=1-\eta_c,C_{\text{BSC}}(e)$. Beneficially, if $\eta_c < 1$, (true in practice, see LDPC figure), then as $e$ goes to zero (it does in higher slices), there's always a nonzero leakage floor $1-\eta_c$ instead of zero leakage, which prevents $\beta$ from diverging to infinity. It 'punishes' too high number of slices, as is appropriate.




---

## 1) General bookkeeping

Let $T(X)$ be the binary label(s) produced by your reconciliation mapper (slices or an MD bit). Let $n$ be the blocklength but we work per raw sample.

- **bits_sent** = the entropy of the label(s) you aim to keep:
  $$
  \mathbf{bits\_sent} = H\big(T(X)\big)\quad\text{(per raw sample)}.
  $$
  For multi‑slice SR, $T(X)\in\{0,1\}^m$ and $H(T)\le m$. For MD with one bit, typically $H(T)\approx 1$ if balanced.

- **bits_leaked** = the fraction of disclosed parity/syndrome bits:
  $$
  \mathbf{bits\_leaked} =
  \begin{cases}
    \sum_i \text{LEAK}_{\rm EC}(e_i,\eta_{c,i}) & \text{(SR)}\\
    \text{LEAK}_{\rm EC}(e,\eta_c) & \text{(MD)}
  \end{cases}
  $$

- **Denominator** ($I_{AB}$) is the mutual information between the analog variables actually used for reconciliation (precisely specified per setting below). With no sifting, $I_{AB}=I(X;Y)$. With guard bands, use the conditional form $I(X;Y\mid A=1)$.

The efficiency is then
$$
\boxed{\beta=\dfrac{H(T(X)) - \text{bits\_leaked}}{I_{AB}}.}
$$

> Balanced‑slice remark. Modeling each slice as BSC implicitly treats slice bits as roughly balanced. If they are not, replace $h_2(e)$ by $H(S\mid\tilde S)$ per slice. We keep the BSC approximation here by design.

---

## 2) Sliced Reconciliation (SR, multilevel, one‑way LDPC)

**Setup.** Quantize to $m$ binary slices $(S_1,\dots,S_m)$. For slice $i$, let estimated BER be $e_i$ under the chosen hard‑decision detector, and pick an LDPC with efficiency $\eta_{c,i}$.

- **bits_sent**: $H\big(T(X)\big) = H\big([S_1,\dots,S_m]\big)$.
- **bits_leaked**: $\displaystyle \sum_{i=1}^m \big(1-\eta_{c,i}\big(1-h_2(e_i)\big)\big)$.
- **Denominator**: $I_{AB}=I(X;Y)$ (no sifting assumed here).

**Efficiency.**
$$
\boxed{\beta_{\rm SR} = \dfrac{H\big([S_1,\dots,S_m]\big) - \sum_{i=1}^m \big(1-\eta_{c,i}(1-h_2(e_i))\big)}{I(X;Y)}.}
$$

> Uniform‑$\eta_c$ simplification. If $\eta_{c,i}=\eta_c$ for all $i$, then $\text{bits\_leaked}= m-\eta_c\sum_i (1-h_2(e_i))$.

---

## 3) Guard‑Band Sliced Reconciliation (GBSR)

**Setup.** Introduce a pass indicator $A\in\{0,1\}$. Keep only samples with $A=1$. Let $p_{\rm pass}=\Pr[A=1]$. All statistics below are conditional on passing.

- **bits_sent** (per kept sample): $H(T(X)\mid A=1)$.
- **bits_leaked** (per kept sample): $\displaystyle \sum_i \big(1-\eta_{c,i}(1-h_2(e_i\mid A=1))\big)$.
- **Denominator**: $I_{AB}=I(X;Y\mid A=1)$.

**Efficiency (per kept sample).**
$$
\boxed{\beta_{\rm GBSR} = \dfrac{H\big(T(X)\mid A=1\big) - \sum_i \big(1-\eta_{c,i}(1-h_2(e_i\mid A=1))\big)}{I(X;Y\mid A=1)}.}
$$

Throughput per raw sample: if you want efficiency multiplied by survival probability, multiply numerator and denominator by $p_{\rm pass}$. The ratio above stays the same; total delivered secret bits per raw sample picks up the factor $p_{\rm pass}$.

---

## 4) Multidimensional Reconciliation (MD, one‑way LDPC)

**Setup.** Map blocks of analog symbols to a single binary label $T\in\{0,1\}$ (e.g., 2D/4D sign mapping). Use hard decisions to estimate a BSC BER $e_{\rm MD}$. Choose LDPC with efficiency $\eta_c$ for that BER.

- **bits_sent**: $H(T(X))$ (often $\approx 1$ if balanced).
- **bits_leaked**: $\text{LEAK}_{\rm EC}(e_{\rm MD},\eta_c)=1-\eta_c\big(1-h_2(e_{\rm MD})\big)$.
- **Denominator**: $I_{AB}=I(X;Y)$ for the MD‑transformed variables you reconcile.

**Efficiency.**
$$
\boxed{\beta_{\rm MD} = \dfrac{H\big(T(X)\big) - \big(1-\eta_c(1-h_2(e_{\rm MD}))\big)}{I(X;Y)}.}
$$

> If the MD mapping is followed by a guard‑band, use the GBSR conditioning recipe: replace all quantities by their $(A=1)$ counterparts and take $I_{AB}=I(X;Y\mid A=1)$.

---

## 5) Implementation checklist (hard‑decision, one‑way only)

1. Decide SR vs MD (and whether guard bands are used).
2. Estimate BER(s): $e_i$ for SR slices, or $e_{\rm MD}$ for MD, **after** all preprocessing/detectors you will actually use.
3. Pick $\eta_c$ (per slice if needed) from the real code design at those BERs.
4. Compute **bits_sent** = entropy of the label(s) ($H(T)$) (empirical if needed).
5. Compute **bits_leaked** via $1-\eta_c(1-h_2(e))$ and sum over slices as appropriate.
6. Set the denominator ($I_{AB}$) to the mutual information of the exact variables used for reconciliation (conditional on $A=1$ if guard‑banded).
7. Return $\beta$ from the boxed formulas above.

---

## 6) Sanity checks

- As $e\to 0$ and you adapt to high‑rate codes ($\eta_c\to 1$), $\text{LEAK}_{\rm EC}\to 0$ and $\beta\to H(T)/I_{AB}$.
- As $e\to \tfrac12$, $h_2(e)\to 1$, $R\to 0$, $\text{LEAK}_{\rm EC}\to 1$, so the numerator collapses, $\beta\to 0$.
- For SR with identical slices, the expression recovers the usual $H(Q)-m+\sum R_i$ numerator because $H(Q)=H(T)$ and $\sum(1-R_i)=\sum \text{LEAK}_{\rm EC}(e_i,\eta_{c,i})$.

---

### What we are not modeling here

No soft information, no iterative/interactive protocols (Cascade/Winnow), no syndrome header/CRC overheads, no finite‑length rate loss, and no slice‑bias corrections. Add those as extra leak terms if you include them in practice.