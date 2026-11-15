# tsui-clyne

# Overview
Object oriented Python implementation of the Tsui-Clyne analytical model for residual stresses in progressively deposited coatings. The package currently implements the framework for *planar geometries* according to:

> Y. C. Tsui and T. W. Clyne, *An analytical model for predicting residual stresses in progressively deposited coatings. Part 1: Planar geometry*, Thin Solid Films **306** (1997) 23–33.


## What it computes

- **Quenching stress**, pass by pass: position of the neutral axis, composite bending stiffness and curvature increment after each pass.
- **Thermal mismatch stress** from cooling, driven by the misfit strain $\Delta \varepsilon = \Delta \alpha \cdot \Delta T$ (CTE mismatch between substrate and deposit).
- **Final stress levels** in the substrate (bottom face and interface) and in the midpoint of each deposit layer.
- **Stress after substrate removal**: the force and moment carried by the substrate are redistributed onto the deposit as a linear stress profile, which is subtracted from the as-deposited profile.

## Inputs
| Symbol | Meaning |
|---|---|
| $E_s,\ \nu_s$ | Substrate Young's modulus (GPa) and Poisson's ratio |
| $E_d,\ \nu_d$ | Deposit Young's modulus (GPa) and Poisson's ratio |
| $H$ | Substrate thickness |
| $w$ | Thickness of a single pass |
| $n$ | Number of passes (total deposit thickness $h = n\,w$) |
| $\sigma_q$ | Quench stress |
| $b$ | Width (stresses do not depend on it for a uniform section) |
| $\Delta\varepsilon$ | Thermal misfit strain, $(\alpha_s-\alpha_d)\,\Delta T$ |
Effective moduli for the biaxial stress state is computed as follows:
$$
E_{\text{eff}} = \frac{E}{1-\nu}
$$

## Usage

```python
from tc import TsuiClyne   # adapt to your module layout

tc = TsuiClyne(...)        # materials, geometry, sigma_q, delta_eps
tc.deposition()            # quench build-up, substrate and deposit stresses
```