Here’s a tight checklist you can turn into prose. Short, declarative, and complete.

* **Channel & notation (heterodyne).**
  Dual-homodyne (heterodyne) measurement per real dimension: $Y=\sqrt{T/2}\,X+Z$, with $X\sim\mathcal N(0,V_{\rm mod})$, $Z\sim\mathcal N(0,1+\xi/2)$.
  SNR $=\dfrac{(T/2)\,V_{\rm mod}}{1+\xi/2}$.
  Use the standard EB covariance for $\chi_{EB}(T,\xi,V_{\rm mod})$.

* **Quantizer without equiprobability.**
  Thresholds $\boldsymbol\tau\!: -\infty=\tau_0<\cdots<\tau_K=\infty$ define cells $I_k$.
  Cell masses $p_k(\boldsymbol\tau)=\int_{\tau_{k-1}}^{\tau_k}\phi_{0,V_{\rm mod}}(x)\,dx$.
  Label $Q\in\{1,\dots,K\}$ with entropy $H(Q)=-\sum_k p_k\log_2 p_k$ (not forced to equal $m$).
  Posterior $X|Y=y\sim\mathcal N(\mu_{X|y},\sigma^2_{X|Y})$ with
  $\mu_{X|y}=\dfrac{\sqrt{T/2}\,V_{\rm mod}}{V_Y}y$, $\sigma^2_{X|Y}=V_{\rm mod}-\dfrac{(T/2)V_{\rm mod}^2}{V_Y}$, $V_Y=\tfrac{T}{2}V_{\rm mod}+1+\tfrac{\xi}{2}$.
  Cell posteriors $p(k|y)=\Phi\!\Big(\dfrac{\tau_k-\mu_{X|y}}{\sigma_{X|Y}}\Big)-\Phi\!\Big(\dfrac{\tau_{k-1}-\mu_{X|y}}{\sigma_{X|Y}}\Big)$.
  Conditional entropy $H(Q|Y)=-\!\int \phi_{0,V_Y}(y)\sum_k p(k|y)\log_2 p(k|y)\,dy$.
  Mutual information $I(Q;Y)=H(Q)-H(Q|Y)$.

* **Single coding knob ($\beta_c$).**
  $\beta_c\in(0,1]$ is the code efficiency used for all schemes.
  It maps to a scheme-specific $f_{\rm EC}$ that multiplies the Slepian–Wolf term.

* **SR key rate (no post-selection).**
  $f_{\rm EC}^{\rm SR}=1+(1-\beta_c)\,\dfrac{I(Q;Y)}{H(Q|Y)}$.
  $K_\infty^{\rm SR}=H(Q)-f_{\rm EC}^{\rm SR}\,H(Q|Y)-\chi_{EB}=\beta_c\,I(Q;Y)-\chi_{EB}$.
  Overall reconciliation efficiency: $\beta_{\rm SR}=\dfrac{K_\infty^{\rm SR}+\chi_{EB}}{I_{AB}}=\beta_c\,\dfrac{I(Q;Y)}{I_{AB}}$.
  **To optimize:** $V_{\rm mod}$ and $\boldsymbol\tau$.

* **ORSR key rate (guard-band post-selection).**
  Define acceptance set $\mathcal A\subset\mathbb R$ via guard widths; $p_{\rm pass}=\int_{\mathcal A}\phi_{0,V_Y}(y)\,dy$.
  Post-selected conditional entropy $H(Q|Y_{\rm PS})=-\!\int_{\mathcal A}\!\frac{\phi_{0,V_Y}(y)}{p_{\rm pass}}\sum_k p(k|y)\log_2 p(k|y)\,dy$.
  $I(Q;Y_{\rm PS})=H(Q)-H(Q|Y_{\rm PS})$.
  $f_{\rm EC}^{\rm ORSR}=1+(1-\beta_c)\,\dfrac{I(Q;Y_{\rm PS})}{H(Q|Y_{\rm PS})}$.
  **Exact DW with sifting (per channel use):**
  $K_\infty^{\rm ORSR}=p_{\rm pass}\big[H(Q)-f_{\rm EC}^{\rm ORSR}H(Q|Y_{\rm PS})-\chi_{EB}^{\rm PS}\big]$.
  **Two practical choices for $\chi$:**
  — Optimistic: $\chi_{EB}^{\rm PS}\approx \chi_{EB}^{\rm G}$ and keep it **inside** the bracket.
  — Conservative: use $p_{\rm pass}[H(Q)-f_{\rm EC}^{\rm ORSR}H(Q|Y_{\rm PS})]-\chi_{EB}^{\rm G}$ (take $\chi$ **outside**).
  “β per raw use”: $\tilde\beta_{\rm ORSR}=p_{\rm pass}\,\beta_c\,\dfrac{I(Q;Y_{\rm PS})}{I_{AB}}$.
  **To optimize:** $V_{\rm mod}$, $\boldsymbol\tau$, guard widths.

* **MD key rate (virtual BIAWGNC).**
  Effective SNR $s=\dfrac{(T/2)V_{\rm mod}}{1+\xi/2}$.
  Capacity $C_{\rm BIAWGNC}(s)=1-\int \phi(z)\log_2\big(1+e^{-2\sqrt{s}\,z-2s}\big)\,dz$.
  $f_{\rm EC}^{\rm MD}(s)=\dfrac{1-\beta_c\,C(s)}{1-C(s)}=1+(1-\beta_c)\dfrac{C(s)}{1-C(s)}$.
  $K_\infty^{\rm MD}=1-f_{\rm EC}^{\rm MD}(1-C(s))-\chi_{EB}=\beta_c\,C(s)-\chi_{EB}$.
  $\beta_{\rm MD}=\beta_c\,\dfrac{C(s)}{I_{AB}}$.
  **To optimize:** $V_{\rm mod}$.

* **Interpretation (how $\beta$ and $f_{\rm EC}$ relate).**
  Devetak–Winter form: $K_\infty=\beta\,I_{AB}-\chi_{EB}$.
  For SR: $\beta=\beta_c\,\dfrac{I(Q;Y)}{I_{AB}}$ and $f_{\rm EC}^{\rm SR}$ is the exact multiplier that turns $H(Q|Y)$ into the coded leak.
  For ORSR: same, but replace $I(Q;Y)$ by $p_{\rm pass}\,I(Q;Y_{\rm PS})$ for a per-use efficiency.
  For MD: $\beta=\beta_c\,\dfrac{C_{\rm BIAWGNC}(s)}{I_{AB}}$; the leak is $f_{\rm EC}^{\rm MD}(1-C)$.

* **Takeaway for implementation.**
  Use one $\beta_c$ across SR/ORSR/MD.
  Compute $H(Q)$, $H(Q|Y)$ (or $H(Q|Y_{\rm PS})$) from the 1-D $y$ integral; never assume equiprobability.
  Plug those into the formulas above; optimize $V_{\rm mod}$, $\boldsymbol\tau$ (and guards).
  State clearly whether $\chi$ is post-selected and inside the bracket, or a conservative outside-bracket Gaussian value.
