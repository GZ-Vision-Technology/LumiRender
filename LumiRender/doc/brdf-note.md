# BRDF optimization node

## Beckmann and TrowBridgeReitz BRDF optimization
 ### Normal distrubiton
   $$
    \begin{aligned}
    & \left( \frac{\cos\phi}{\alpha_x} \right)^2 + \left( \frac{\sin\phi}{\alpha_y} \right)^2 \\
    = & \left( \frac{1}{\alpha_x^2} - \frac{1}{\alpha_y^2} \right)\cos^2\phi + \frac{1}{\alpha_y^2} \\
    = & \left( \frac{1}{\alpha_x^2} - \frac{1}{\alpha_y^2} \right) \frac{x^2_h}{1 - z_h^2} + \frac{1}{\alpha_y^2}
    \end{aligned}
  $$
 ### Shadowing term
 $$
  \cos^2\phi\alpha_x^2 + \sin^2\phi\alpha_y^2 = (\alpha_x^2-\alpha_y^2)\cos^2\phi + \alpha_y^2
 $$

## Ashikhmin BRDF
  ### Normal distribution
  $$
   D(\omega_h) = c\left(  1 + 4\frac{e^{-\frac{\cot^2\theta_h}{\alpha^2}}}{\sin^4\theta_h} \right)
  $$

  $$
  \begin{aligned}
    & 2\pi\int_0^{\pi/2} \left(1 + 4e^{-\cot^2\theta/\alpha^2}/\sin^4\theta \right)  \sin\theta\cos\theta d\theta \\
   = & 2\pi\left( \frac12 -2 \int_0^{\pi/2} e^{-\cot^2\theta/\alpha^2} d \cot^2\theta \right) \\
   = \pi\left( 1 + 4\alpha^2e^{-\cot^2\theta/\alpha^2}\big|_0^{\pi/2} \right)
  \end{aligned}
  $$
  $$
   c = \frac{1}{\pi(1 + 4\alpha^2)}
  $$
  importance sample:
  $$
    c\pi(1 + 4\alpha^2e^{-\cot^2\theta/\alpha^2}) = \xi_2
  $$
  $$
    \cot^2\theta = -4\alpha^2\ln\frac{1}{4\alpha^2}\left(\frac{\xi_2}{c\pi} - 1\right)
  $$


## Neubelt cloth BRDF
  ### Normal  distribution
  Isotripic
  $$
   D(\omega_h) = \frac{1}{2\pi}\left(2 + \frac{1}{\alpha}\right) \sin^{1/\alpha}\theta_h
  $$

  ### Layering
  Specular albedo should be precomputed using
  $$
  \begin{aligned}
      \alpha_i = \alpha(\omega_i) = &\int_{\mathcal{H}^2} f_{r,spec}\cdot \cos\theta_o d\omega_o \\
        \alpha_o = \alpha(\omega_o) =& \int_{\mathcal{H}^2} f_{r,spec} \cdot \cos\theta_i d\omega_i 
  \end{aligned}
  $$
  Average albedo of the specular component is:
  $$
    \alpha_{spec}^{avg} = \frac{1}{\pi} \int_{\mathcal{H}^2} \alpha_(\omega)\cos\theta d\omega
  $$

  Layered BRDF is
  $$
   \begin{aligned}
     f_r = & f_{r,matte} + f_{r,spec} \\
         = & k(\lambda)\frac{(1-\alpha_o)(1-\alpha_i)}{\pi(1 - \alpha_{spec}^{avg})} + \frac{D(\omega_h)G(\omega_o,\omega_i)F(\omega_h,\omega_i)}{4(n \cdot \omega_o)(n \cdot \omega_i)}
   \end{aligned}
  $$


  $$
    d\omega_h = \sin\theta_hd\theta_hd\phi_h \\
    d\omega_i = \sin\theta_id\theta_id\phi_i = \sin2\theta_h2d\theta_hd\phi_h
  $$

  $$
    p(\omega_i) = p(\omega_h)\frac{d\omega_h}{d\omega_i} = p(\omega_h)\frac{1}{4\cos\theta_h}
  $$

