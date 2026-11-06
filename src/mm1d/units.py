"""Unit constants and conversions -- the single conversion point of mm1d.

Internally the library uses SI everywhere: depth ``m``, layer thickness ``m``,
time ``s``, frequency ``Hz``, velocity ``m/s``, density ``kg/m^3``, slowness
``s/m``.  Angles are **degrees at the public API** and radians internally;
degree/radian (and second/millisecond, should a UI layer ever appear)
conversions happen only through this module, so conversion factors cannot
scatter through the code base.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["MS_PER_S", "S_PER_MS", "deg_to_rad", "rad_to_deg"]

MS_PER_S: float = 1.0e3
"""Milliseconds per second; multiply a time in s by this to obtain ms."""

S_PER_MS: float = 1.0e-3
"""Seconds per millisecond; multiply a time in ms by this to obtain s."""


def deg_to_rad(angle_deg: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Convert degrees to radians.

    Parameters
    ----------
    angle_deg : float or ndarray
        Angle in degrees (scalar or array).

    Returns
    -------
    float or ndarray
        Angle in radians.
    """
    return np.asarray(angle_deg, dtype=np.float64) * (np.pi / 180.0)


def rad_to_deg(angle_rad: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Convert radians to degrees.

    Parameters
    ----------
    angle_rad : float or ndarray
        Angle in radians (scalar or array).

    Returns
    -------
    float or ndarray
        Angle in degrees.
    """
    return np.asarray(angle_rad, dtype=np.float64) * (180.0 / np.pi)
