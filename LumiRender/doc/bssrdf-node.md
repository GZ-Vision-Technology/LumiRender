# BSSRDF theory note

## Diffuse Approximate
  Intro-differential form of tansfer equation:
  $$
    \frac{\partial}{\partial t} L_o(p + t\omega, \omega) = -\sigma_t(p, \omega) L_i(p, \omega)
     + \sigma_s(p,\omega)\int_{\mathcal{S}^2} p(p, -\omega',\omega) L_i(p,\omega') d\omega' + L_e(p, \omega) \tag{1.1}
  $$
  Assume spatially uniform material parameters and isotropic phace function $p:=1/4\pi$, replace $L_o(p,\omega) = L_i(p,-\omega)$, $L(p,\omega) := L_i(p,\omega)$, then
  $$
   \frac\partial{\partial t} L(p+t\omega,\omega) = -\sigma_t(p,\omega) L(p,\omega) + \frac{\sigma_s(p,\omega)}{4\pi}\int_{\mathcal{S}^2} L(p,\omega') d\omega' + L_e(p,\omega). \tag{1.2}
  $$

  For iraddiance quantity $L_d(p, \cdot)$, its two-term expansion approximate is:
  $$
   L_d(p,\cdot) = \frac{1}{2\pi}\phi(p) + \frac{3}{4\pi}\omega\cdot \mathbf{E}(p) \tag{1.3}
  $$
  where
  $$
  \begin{cases}
    \phi(p) = \mu_0[L_d(p,\cdot)] &:= \int_{\mathcal{S}^2} L_d(p,\cdot)d\omega, \text{(radiance fluence)}\\
    \mathbf{E}(p) = \mu_1{L_d(p,\cdot)} &:= \int_{\mathcal{S}^2} \omega\cdot L_d(p,\cdot)d\omega \text{(vector irradiance)}
  \end{cases}
  $$

  Enforce equality of moments in eq (1.1), eq.
  $$
  \mu_i\left[\frac\partial{\partial t} L(p+t\omega,\omega) \right] = \mu_i\left[ -\sigma_t(p,\omega) L(p,\omega) + \frac{\sigma_s(p,\omega)}{4\pi}\int_{\mathcal{S}^2} L(p,\omega') d\omega' + L_e(p,\omega)\right]. \tag{1.2}
  $$
  for $i=0$, the end result is
  $$
   \nabla \mathbf{E}(p) = -(\sigma_t' + \sigma_s')\phi(p) = -\sigma_a\phi(p) + \mathbf{Q}_0(p)
  $$
  where $\mathbf{Q}_i(p) = \mu_i[L_e(p,\cdot)]$.
  