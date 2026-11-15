import os
import sys
sys.path.append('../../src/')

import numpy as np


def th_neutral_axis(H: float, E_s: float, h: float, E_d: float) -> float:
    """Calculate the neutral axis position under thermal loading.

    Parameters
    ----------
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    h : float
        Deposit height.
    E_d : float
        Deposit Young's modulus.

    Returns
    -------
    float
        Neutral axis position measured from the substrate-deposit interface.
    """
    return (E_d * (h**2) - E_s * (H**2)) / (2 * (E_d * h + E_s * H))


def th_curvature_diff(H: float, E_s: float, h: float, E_d: float, delta_eps: float) -> float:
    """Calculate the thermal curvature increment.

    Parameters
    ----------
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    h : float
        Deposit height.
    E_d : float
        Deposit Young's modulus.
    delta_eps : float
        Difference in thermal strain between the substrate and deposit.

    Returns
    -------
    float
        Thermal curvature increment.
    """
    num = 6 * E_d * E_s * h * H * (h + H) * delta_eps
    D_1 = (E_d**2) * (h**4)
    D_2 = 4 * E_d * E_s * (h**3) * H
    D_3 = 6 * E_d * E_s * (h**2) * (H**2)
    D_4 = 4 * E_d * E_s * h * (H**3)
    D_5 = (E_s**2) * (H**4)
    return num / (D_1 + D_2 + D_3 + D_4 + D_5)


def th_stiffness(b: float, H: float, E_s: float, h: float, E_d: float, th_delta: float) -> float:
    """Calculate the composite bending stiffness under thermal loading.

    Parameters
    ----------
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    h : float
        Deposit height.
    E_d : float
        Deposit Young's modulus.
    th_delta : float
        Neutral axis position under thermal loading.

    Returns
    -------
    float
        Composite bending stiffness.
    """
    par = lambda hh, sign: hh**2 / 3 + sign * hh * th_delta + th_delta**2
    return b * E_d * h * par(h, -1) + b * E_s * H * par(H, 1)


def force_cte(H: float, h: float, th_k_diff: float, th_sigma: float) -> float:
    """Calculate the equivalent force associated with thermal expansion.

    Parameters
    ----------
    H : float
        Substrate height.
    h : float
        Deposit height.
    th_k_diff : float
        Thermal curvature increment.
    th_sigma : float
        Composite bending stiffness under thermal loading.

    Returns
    -------
    float
        Equivalent thermal force.
    """
    return 2 * th_k_diff * th_sigma / (h + H)


def th_stress_substrate_bottom(b: float, H: float, E_s: float, th_force: float, th_delta: float, th_k_diff: float) -> float:
    """Calculate the thermal stress at the bottom surface of the substrate.

    Parameters
    ----------
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    th_force : float
        Equivalent thermal force.
    th_delta : float
        Neutral axis position under thermal loading.
    th_k_diff : float
        Thermal curvature increment.

    Returns
    -------
    float
        Thermal stress at the substrate bottom surface.
    """
    return -th_force / (b * H) + E_s * th_k_diff * (H + th_delta)


def th_stress_substrate_interface(b: float, H: float, E_s: float, th_force: float, th_delta: float, th_k_diff: float) -> float:
    """Calculate the thermal stress at the substrate-deposit interface.

    Parameters
    ----------
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    th_force : float
        Equivalent thermal force.
    th_delta : float
        Neutral axis position under thermal loading.
    th_k_diff : float
        Thermal curvature increment.

    Returns
    -------
    float
        Thermal stress at the substrate-deposit interface.
    """
    return -th_force / (b * H) + E_s * th_k_diff * th_delta


def th_stress_layer(j: int, b: float, w: float, h: float, E_d: float, th_force: float, th_delta: float, th_k_diff: float) -> float:
    """Calculate the thermal stress within a deposited layer.

    Parameters
    ----------
    j : int
        Deposition step index.
    b : float
        Material width.
    w : float
        Deposit layer thickness.
    h : float
        Deposit height.
    E_d : float
        Deposit Young's modulus.
    th_force : float
        Equivalent thermal force.
    th_delta : float
        Neutral axis position under thermal loading.
    th_k_diff : float
        Thermal curvature increment.

    Returns
    -------
    float
        Thermal stress within the deposited layer.
    """
    return th_force / (b * h) - E_d * th_k_diff * ((j - 0.5) * w - th_delta)