import os
import sys
sys.path.append('../../src/')

from dataclasses import dataclass
from dataclasses import dataclass, field

import numpy as np
from tsuiclyne.util import dict_to_h5

def E_eff(E: float, nu: float) -> float:
    """Calculate the effective Young's modulus.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    float
        Effective Young's modulus, calculated as E / (1 - nu).
    """
    return E / (1 - nu)


@dataclass
class Material:
    """Stores material properties.

    Coordinate convention
    ---------------------
    x : Longitudinal axis (length).
    z : Transverse axis (depth).
    y : Vertical axis (height).

    Attributes
    ----------
    name : str
        Material name.
    b : float
        Material width.
    h : float
        Material height (deposit), y-axis. In the original paper, ``h`` and
        ``H`` denote the deposit and substrate heights, respectively.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    alpha : float
        Coefficient of thermal expansion (CTE).
    E_eff : float
        Effective Young's modulus, calculated as ``E / (1 - nu)``.
    """
    name: str
    b: float
    h: float
    E: float
    nu: float
    alpha: float
    E_eff: float = field(init=False)

    def export(self, path: str, prefix=""):
        _dict = {f"{prefix}{k}": v for k, v in vars(self).items()}
        dict_to_h5(path, _dict)

    def __post_init__(self):
        self.E_eff = self.E / (1 - self.nu)


def test_material():
    mat = Material("mat", 1, 10, 100, 0.3, 1e-6)
    print(mat)
    mat.export("/home/ale/Desktop/tsui-clyne/mat.h5", "/tsui-clyne/substrate/")


if __name__ == "__main__":
    test_material()