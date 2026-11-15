import os
import sys
sys.path.append('../../src/')

import numpy as np
from tsuiclyne.util import tail_sum

def force(j: int, sigma_q: float, b: float, H: float, E_s: float, w: float, E_d: float) -> np.ndarray:
    """Calculate the force associated with deposition step(s).

    Parameters
    ----------
    j : int or np.ndarray
        Deposition step indices. Must contain only positive values.
    sigma_q : float
        Applied stress.
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.

    Returns
    -------
    np.ndarray
        Force associated with each deposition step.
    """
    j = np.asarray(j)
    if np.any(j <= 0):
        raise ValueError("'j' must contain only positive values.")
    return sigma_q * b * w * ((H * E_s + (j - 1) * w * E_d) / (H * E_s + j * w * E_d))


def neutral_axis(j: int, H: float, E_s: float, w: float, E_d: float) -> float | np.ndarray:
    """Calculate the neutral axis position after deposition step(s).

    Parameters
    ----------
    j : int
        Deposition step indices.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.

    Returns
    -------
    float or np.ndarray
        Neutral axis position measured from the substrate-deposit interface.
    """
    return (((j * w)**2) * E_d - ((H**2) * E_s)) / (2 * ((j * w) * E_d + H * E_s))


def stiffness(j: int, b: float, H: float, E_s: float, w: float, E_d: float, delta_j: float | np.ndarray) -> float | np.ndarray:
    """Calculate the composite bending stiffness.

    Parameters
    ----------
    j : int
        Deposition step indices.
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.
    delta_j : float or np.ndarray
        Neutral axis position at deposition step(s).

    Returns
    -------
    float or np.ndarray
        Composite bending stiffness.
    """
    j = np.asarray(j)
    # delta_j = neutral_axis(j, H, E_s, w, E_d)
    A_1 = E_d * b * j * w
    A_2 = ((j * w)**2) / 3 - j * w * delta_j + delta_j ** 2
    A_3 = E_s * b * H
    A_4 = (H**2) / 3 + H * delta_j + delta_j ** 2
    return A_1 * A_2 + A_3 * A_4


def curvature_diff(j: int, w: float, F_j: float | np.ndarray,
                   delta_j_prev: float | np.ndarray, sigma_j: float | np.ndarray) -> float | np.ndarray:
    """Calculate the curvature increment at deposition step(s).

    Parameters
    ----------
    j : int
        Deposition step indices.
    w : float
        Deposit layer thickness.
    F_j : float or np.ndarray
        Force associated with deposition step(s).
    delta_j_prev : float or np.ndarray
        Neutral axis position at the preceding deposition step(s).
    sigma_j : float or np.ndarray
        Composite bending stiffness at deposition step(s).

    Returns
    -------
    float or np.ndarray
        Curvature increment.
    """
    j = np.asarray(j)
    # F_j = force(j, sigma_q, b, H, E_s, w, E_d)
    # sigma_j = stiffness(j, b, H, E_s, w, E_d)
    # delta_j_prev = neutral_axis(j-1, H, E_s, w, E_d)
    C = (j - 0.5) * w - delta_j_prev
    return F_j * C / sigma_j


def stress_substrate_common(steps: int, b: float, H: float, E_s: float,
                            w: float, E_d: float, F: float | np.ndarray) -> float | np.ndarray:
    """Calculate the common axial stress contribution in the substrate.

    Parameters
    ----------
    steps : int
        Total number of deposition steps.
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.
    F : float or np.ndarray
        Applied force.

    Returns
    -------
    float or np.ndarray
        Common substrate stress contribution.
    """
    return -E_s * F / (b * (H * E_s + (steps - 1) * w * E_d))


def stress_substrate_bottom(steps: int, b: float, H: float, E_s: float,
                            w: float, E_d: float,
                            F: float | np.ndarray, delta: float | np.ndarray, k_diff: float | np.ndarray) -> float | np.ndarray:
    """Calculate the stress at the bottom surface of the substrate.

    Parameters
    ----------
    steps : int
        Total number of deposition steps.
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.
    F : float or np.ndarray
        Applied force.
    delta : float or np.ndarray
        Neutral axis position.
    k_diff : float or np.ndarray
        Curvature increment.

    Returns
    -------
    float or np.ndarray
        Stress at the substrate bottom surface.
    """
    return stress_substrate_common(steps, b, H, E_s, w, E_d, F) + E_s * k_diff * (H + delta)


def stress_substrate_interface(steps: int, b: float, H: float, E_s: float,
                               w: float, E_d: float, F: float | np.ndarray,
                               delta: float | np.ndarray, k_diff: float | np.ndarray) -> float | np.ndarray:
    """Calculate the stress at the substrate-deposit interface.

    Parameters
    ----------
    steps : int
        Total number of deposition steps.
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.
    F : float or np.ndarray
        Applied force.
    delta : float or np.ndarray
        Neutral axis position.
    k_diff : float or np.ndarray
        Curvature increment.

    Returns
    -------
    float or np.ndarray
        Stress at the substrate-deposit interface.
    """
    return stress_substrate_common(steps, b, H, E_s, w, E_d, F) + E_s * k_diff * delta


def stress_layer(steps: int, b: float, w: float, E_d: float,
                 F: float | np.ndarray, delta: float | np.ndarray, k_diff: float | np.ndarray) -> float | np.ndarray:
    """Calculate the stress distribution within a deposited layer.

    Parameters
    ----------
    steps : int
        Total number of deposition steps.
    b : float
        Material width.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.
    F : float or np.ndarray
        Applied force.
    delta : float or np.ndarray
        Neutral axis position.
    k_diff : float or np.ndarray
        Curvature increment.

    Returns
    -------
    float or np.ndarray
        Stress distribution within the deposited layer(s).
    """
    return (F / (b * w)) - E_d * k_diff * ((steps - 0.5) * w - delta)


def stress_deposit_layer(j: int, steps: int, b: float,
                         H: float, E_s: float,
                         w: float, E_d: float,
                         F: float | np.ndarray, delta: float | np.ndarray, k_diff: np.ndarray) -> float | np.ndarray:
    """Calculate the total stress in a deposited layer.

    Parameters
    ----------
    j : int
        Index of the deposited layer(s).
    steps : int
        Total number of deposition steps.
    b : float
        Material width.
    H : float
        Substrate height.
    E_s : float
        Substrate Young's modulus.
    w : float
        Deposit layer thickness.
    E_d : float
        Deposit Young's modulus.
    F : float or np.ndarray
        Applied force.
    delta : float or np.ndarray
        Neutral axis position.
    k_diff : np.ndarray
        Curvature increments.

    Returns
    -------
    float or np.ndarray
        Total stress in the specified deposited layer(s).
    """
    j = np.asarray(j)
    own = stress_layer(steps, b, w, E_d, F, delta, k_diff)[j - 1]
    t_force = -E_d * F / (b * (H * E_s + (steps - 1) * w * E_d))
    S_force = tail_sum(t_force, j)
    S_curv = -E_d * ((j - 0.5) * w  * tail_sum(k_diff, j) - tail_sum(k_diff * delta, j))
    return own + S_force + S_curv


def force_resultant(b: float, H: float, slope: float, intercept: float) -> float:
    """Calculate the force resultant over the substrate thickness.

    Parameters
    ----------
    b : float
        Material width.
    H : float
        Substrate height.
    slope : float
        Slope of the linear stress distribution.
    intercept : float
        Stress value at y = 0.

    Returns
    -------
    float
        Resultant force obtained by integrating the stress distribution
        over the substrate cross-section (rectangle).
    """
    return b * (intercept * H - slope * H**2 / 2)


def moment_resultant(b: float, H: float, slope: float, intercept: float) -> float:
    """Calculate the moment resultant about the interface (y = 0).

    Parameters
    ----------
    b : float
        Material width.
    H : float
        Substrate height.
    slope : float
        Slope of the linear stress distribution.
    intercept : float
        Stress value at y = 0.

    Returns
    -------
    float
        Resultant bending moment about y = 0.
    """
    return b * (slope * H**3 / 3 - intercept * H**2 / 2)


def redistribute(b: float, h: float, F_s: float, M_s: float) -> tuple[float, float]:
    """Calculate the linear stress distribution required in the deposit
    to balance the substrate force and moment resultants.

    The redistributed stress is expressed as A + B*z, where z ranges
    from 0 to h. The deposit resultants satisfy F_d = -F_s and M_d = -M_s.

    Parameters
    ----------
    b : float
        Material width.
    h : float
        Deposit height.
    F_s : float
        Substrate force resultant.
    M_s : float
        Substrate moment resultant.

    Returns
    -------
    tuple[float, float]
        Slope (B) and intercept (A) of the linear stress distribution.
        The intercept corresponds to the stress at z = 0.
    """
    F_d, M_d = -F_s, -M_s
    slope     = 12 * M_d / (b * h**3) - 6 * F_d / (b * h**2)
    intercept =  4 * F_d / (b * h)    - 6 * M_d / (b * h**2)
    return slope, intercept