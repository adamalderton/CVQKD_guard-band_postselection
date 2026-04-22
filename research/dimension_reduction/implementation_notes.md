# Figure Equations Implemented in `dim_red_dev.py`

This note documents the equations used to generate:

- `plots/epsilon_total_distance_key_sweep.png`

for

$$
\epsilon_{\mathrm{tot}} \in \{0,\;10^{-6},\;10^{-4},\;10^{-2}\},
$$

with dimension-reduction weight fixed to

$$
W = 0.
$$

## 1) Distance to Transmittance

For each distance $L$ (km), the channel transmittance is

$$
T(L) = 10^{-\alpha L / 10}, \qquad \alpha = 0.2\ \mathrm{dB/km}.
$$

## 2) Per-distance Optimisation

For each $(L,\epsilon_{\mathrm{tot}})$, the script optimises over modulation variance and pass probability:

$$
(V_{\mathrm{mod}}^\star, p_{\mathrm{pass}}^\star)
=
\arg\max_{V_{\mathrm{mod}},\,p_{\mathrm{pass}}}
K_{\mathrm{final}}^{\mathrm{pulse}}(L, \epsilon_{\mathrm{tot}}; V_{\mathrm{mod}}, p_{\mathrm{pass}})
$$

subject to

$$
V_{\mathrm{mod}} \in [0.01,\,10], \qquad
p_{\mathrm{pass}} \in [0.01,\,1].
$$

For each candidate, $\tau$ is chosen equiprobably and guard bands $g$ are generated from $p_{\mathrm{pass}}$.

## 3) Mapping $\epsilon_{\mathrm{tot}} \rightarrow \epsilon_{\phi_G}$ (with $W=0$)

The implementation uses the new epsilon route in `GBSR.evaluate_quantised_key_efficiency_from_epsilon_phi_g`.

The purification distance is

$$
\epsilon_{\mathrm{pur}}
=
\sqrt{2\epsilon_{\phi_G} - \epsilon_{\phi_G}^2}.
$$

With $W=0$, truncation terms are zero in this figure, so

$$
\epsilon_{\mathrm{cq}} = \epsilon_{\mathrm{pur}} = \epsilon_{\mathrm{tot}}.
$$

Hence $\epsilon_{\phi_G}$ is computed from $\epsilon_{\mathrm{tot}}$ by inverting the above relation:

$$
\epsilon_{\phi_G}
=
1 - \sqrt{1-\epsilon_{\mathrm{tot}}^2}
=
\frac{\epsilon_{\mathrm{tot}}^2}{1+\sqrt{1-\epsilon_{\mathrm{tot}}^2}},
$$

and the second form is used for numerical stability.

## 4) Continuity Correction

The continuity helper is

$$
\kappa(\epsilon, |\mathcal{X}|) =
\epsilon \log_2 |\mathcal{X}|
+
(1+\epsilon)\,
h_2\!\left(\frac{\epsilon}{1+\epsilon}\right),
$$

with binary entropy

$$
h_2(p) = -p\log_2 p -(1-p)\log_2(1-p).
$$

The Holevo continuity penalty per channel use is

$$
\Delta_{\chi} =
2\,\kappa(\epsilon_{\mathrm{cq}}, |\mathcal{UF}|)
+
2\,\kappa(\epsilon_{\mathrm{cq}}, 2).
$$

In conservative mode:

$$
|\mathcal{UF}| = 2|\mathcal{Z}|,\qquad |\mathcal{Z}|=2^m.
$$

For this figure, $m=1$, so $|\mathcal{Z}|=2$ and $|\mathcal{UF}|=4$.

## 5) Base Gaussian Key Rate and Final Rate

The Gaussian key (before new continuity terms) is computed by
`evaluate_quantised_maximum_key_efficiency`:

$$
R_{\mathrm{G}}^{\mathrm{acc}}
=
H_{T_x,\mathrm{acc}}
-
(1+\beta_{\mathrm{ov}})\,H(T_x|T_y)
-
\chi_{\mathrm{G}}.
$$

Then per symbol and per pulse:

$$
R_{\mathrm{G}} = p_{\mathrm{pass}}\,R_{\mathrm{G}}^{\mathrm{acc}},\qquad
K_{\mathrm{G}}^{\mathrm{pulse}} = n_{\mathrm{sym}}\,R_{\mathrm{G}}.
$$

After continuity correction:

$$
R_{\mathrm{after}\,\chi} = R_{\mathrm{G}} - \Delta_{\chi}.
$$

The optional dimension-reduction penalty is

$$
\Delta(W) = \kappa(\epsilon_W, |\mathcal{Z}|),\qquad
\epsilon_W=\sqrt{2W-W^2}.
$$

For this figure, $W=0\Rightarrow \Delta(W)=0$, so

$$
R_{\mathrm{final}} = R_{\mathrm{after}\,\chi},\qquad
K_{\mathrm{final}}^{\mathrm{pulse}} = n_{\mathrm{sym}}\,R_{\mathrm{final}}.
$$

The plotted value is the non-negative key per pulse after optimisation at each distance.
