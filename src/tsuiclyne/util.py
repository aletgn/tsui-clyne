import os
import sys
sys.path.append('../../src/')

from typing import Dict
import numpy as np
from h5py import File, Empty

def tail_sum(a: np.ndarray, j: int) -> float | np.ndarray:
    """Calculate the reverse cumulative sum from index j onwards.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    j : int
        Starting index for the tail sum.

    Returns
    -------
    float or np.ndarray
        Sum of array elements from index j onwards.
    """
    c = np.append(np.cumsum(a[::-1])[::-1], 0.0)
    return c[j]


def line(stress_tot_bottom: float, y_bottom: float,
         stress_tot_interface: float, y_interface: float = 0.) -> tuple[float, float]:
    """Calculate the slope and intercept of a line passing through two points.

    Parameters
    ----------
    stress_tot_bottom : float
        Stress value at the bottom position.
    y_bottom : float
        Vertical coordinate of the bottom position.
    stress_tot_interface : float
        Stress value at the interface position.
    y_interface : float, optional
        Vertical coordinate of the interface position, by default 0.

    Returns
    -------
    tuple[float, float]
        Slope and y-intercept of the fitted line.
    """
    slope = (stress_tot_interface - stress_tot_bottom) / (y_interface - y_bottom)
    intercept = stress_tot_interface - slope * y_interface
    return slope, intercept


def dict_to_h5(path: str, _dict: Dict) -> None:
    """Export a dictionary to an HDF5 file.

    Each dictionary entry is stored as a dataset in the HDF5 file.
    Existing datasets with matching keys are overwritten. Values of
    ``None`` are stored as empty floating-point datasets.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    _dict : Dict
        Dictionary containing the data to export. Keys are used as
        dataset names and values as dataset contents.

    Returns
    -------
    None
    """
    with File(path, "a") as h:
        for k, v in _dict.items():
            if k in h:
                del h[k]
            if v is None:
                print(f"Exporting {k} (None)")
                h.create_dataset(name=k, data=Empty("f"))
            else:
                print(f"Exporting {k}")
                h.create_dataset(name=k, data=v)