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
|Effective moduli | $E_{\text{eff}} = \frac{E}{1-\nu}$

## Basic Usage
### Import Required Modules
```python
from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"), 'path_to_folder_package'))

from h5py import File # To read results.
from tsuiclyne.model import TsuiClaynePlane as TCP
```

### Run Analysis
``` python
# Define material. alpha_* is optional. Required for cooling only.
substrate = Material("Sub", b_sub, H, E_sub, n_sub, alpha_sub)
deposit = Material("Dep"  , b_dep, h, E_dep, n_dep, alpha_dep)

# Instantiate Tsui-Clyne object. Provide either w_step or n_layer
tc = TCP(substrate, deposit, quench_stress, w_step, n_layer, "Sub-Dep")

# Deposition.
tc.deposition()

# Optional. Cooling computes thermal mismatch due to differences in CTE.
# If delta_eps not provided, must choose delta T and alpha.
tc.cooling(delta_eps)

# Optional. Removal computes residual stress after the removal of substrate.
tc.removal()
```

### Export Results
```python
substrate.export('path_to_substrate_file', prefix="tsui-clyne/substrate/")
deposit.export('path_to_deposit_file', prefix="tsui-clyne/deposit/")
tc.export('path_to_tc_file', prefix="tsui-clyne/model/")
```

### Plot Results
```python
substrate = True # Choose whether you want to plot substrate.
h5 = File('path_to_tc_file', "r")
    with h5 as h:
        # substrate.
        H = h["tsui-clyne/substrate/h"][()]
        sb = h["tsui-clyne/model/ssb"][()]
        st = h["tsui-clyne/model/sst"][()]

        # deposit.
        midpoints = h["tsui-clyne/model/midpoints"][:]
        sd = h["tsui-clyne/model/sd"][:]

fig, ax = plt.subplots()
if substrate:
    ax.plot([-H, 0], [sb, st], color="r", label="Substrate")
ax.plot(midpoints, sd, color="b", label="Deposit")
ax.axvline(0, linestyle=":", color="k", alpha=1)
ax.axhline(0, linestyle=":", color="k", alpha=0.5)
ax.set_xlabel(r"Height")
ax.set_ylabel(r"Residual stress [MPa]")
plt.legend(loc="best")
plt.show()
```