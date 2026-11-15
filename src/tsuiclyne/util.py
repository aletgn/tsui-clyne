import os
import sys
sys.path.append('../../src/')

from h5py import File, Empty
from typing import Dict

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