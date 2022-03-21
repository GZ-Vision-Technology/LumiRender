- [Blender Exporter Plugin](#blender-exporter-plugin)
  - [Blender Coordinates Transform](#blender-coordinates-transform)
    - [World CS](#world-cs)
    - [View CS](#view-cs)
    - [Conclusion](#conclusion)

# Blender Exporter Plugin

## Blender Coordinates Transform
### World CS
Right Hand Coordinates System(RH-CS) to Left Hand Coordinates System(LH-CS) tansform gives:
$$
 H = \begin{bmatrix}
  0 & 1 & 0 & 0 \\
  0 & 0 & 1 & 0 \\
  1 & 0 & 0 & 0 \\
  0 & 0 & 0 & 1
 \end{bmatrix} \tag{1.1}
$$
For a Point $P$ in RH-CS, $P'$ in LH-CS is:
$$ P' = HP \tag{1.2} $$
which interprete to:
$$
 P' \triangleq \begin{bmatrix} x'\\y'\\z' \end{bmatrix}
  = \begin{bmatrix} y \\ z \\ x \end{bmatrix} \triangleq P \tag{1.3}
$$

### View CS
Right Hand Camera CS to Left Hand Camera CS:
$$
 \text{LH} \longrightarrow V' = W \cdot V \longleftarrow \text{RH} \tag{1.4}
$$
where:
$$
 W = \begin{bmatrix}
  1 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & -1 & 0 \\
  0 & 0 & 0 & 1
 \end{bmatrix} \tag{1.5}
$$

For a point $P'$ in LH world space, we can deduce that:
$$
 P_\mathrm{v}' = V'P = WVP = WVH^{-1}P' \tag{1.6}
$$
where $H^{-1}$ is reverse of $H$:
$$
 H^{-1} = H^T = \begin{bmatrix}
 0 & 0 & 1 & 0\\
 1 & 0 & 0 & 0\\
 0 & 1 & 0 & 0 \\
 0 & 0 & 0 & 1
 \end{bmatrix} \tag{1.7}
$$
For the simplication, we record:
$$
  P_\mathrm{v}' = V'P',\ \text{where } V' = WVH^{-1}. \tag{1.8}
$$

### Conclusion
To transform 3D coordinates form blender RH-CS to LH-CS, we need the following steps:
 <ol>
  <li> Use eq(1.2) to transform mesh vertices exported form blender. </li>
  <li>  Use eq(1.8) to compute view matrix in LH-CS. </li>
 </ol>
