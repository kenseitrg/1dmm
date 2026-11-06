"""Ricker wavelet source and its spectrum.

The v1 source wavelet for all mm1d wavefield generation.  Two consistent
representations:

- :func:`ricker_wavelet` -- zero-phase time-domain wavelet normalized to unit
  peak amplitude, ``w(0) = 1``.
- :func:`ricker_spectrum` -- its analytic Fourier transform (``e^{-i w t}``
  convention), ``W(f) = 2 f^2 / (sqrt(pi) f_dom^3) * exp(-(f/f_dom)^2)``,
  in seconds.

The two are normalized consistently for sampled data: the numpy ``rfft`` of
the time wavelet sampled at interval ``dt`` equals ``W(f) / dt`` on the FFT
frequency grid (up to negligible aliasing, since the Ricker is effectively
band-limited).  Consequently, convolving a trace with ``ricker_wavelet`` and
multiplying its spectrum by ``ricker_spectrum(f) / dt`` before the inverse
transform are the same operation; the peak amplitude of an isolated event in
the synthetic equals its impulse amplitude (wavelet peak = 1).
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["ricker_spectrum", "ricker_wavelet"]


def ricker_wavelet(
    t_s: float | NDArray[np.float64], f_dom_hz: float
) -> float | NDArray[np.float64]:
    """Zero-phase Ricker wavelet with unit peak amplitude.

    ``w(t) = (1 - 2 pi^2 f_dom^2 t^2) * exp(-pi^2 f_dom^2 t^2)``: symmetric
    about ``t = 0`` with ``w(0) = 1``, side-lobe minima ``-2 exp(-3/2)`` at
    ``t = +/- sqrt(3/2) / (pi f_dom)``, and zero mean.

    Parameters
    ----------
    t_s : float or ndarray
        Time, s (positive or negative; the wavelet is centered at 0).
    f_dom_hz : float
        Dominant frequency, Hz (> 0).

    Returns
    -------
    float or ndarray
        Wavelet amplitude (dimensionless), peak 1 at ``t = 0``.
    """
    if not np.isfinite(f_dom_hz) or f_dom_hz <= 0.0:
        raise ValueError(f"f_dom_hz must be finite and > 0, got {f_dom_hz}")
    t = np.asarray(t_s, dtype=np.float64)
    u = (np.pi * f_dom_hz * t) ** 2
    wavelet = (1.0 - 2.0 * u) * np.exp(-u)
    if t.ndim == 0:
        return wavelet.reshape(-1)[0].item()
    return wavelet


def ricker_spectrum(
    freq_hz: float | NDArray[np.float64], f_dom_hz: float
) -> float | NDArray[np.float64]:
    """Analytic Fourier transform of the unit-peak Ricker wavelet.

    ``W(f) = 2 f^2 / (sqrt(pi) f_dom^3) * exp(-(f/f_dom)^2)`` with the
    ``e^{-i w t}`` transform convention.  Properties: non-negative, peaked at
    ``f = f_dom``; ``W(0) = 0`` (zero-mean wavelet); ``integral of W df = 1``
    (unit peak amplitude).  Units: seconds (integral over time of the
    dimensionless wavelet).

    Combined with a sampled trace: ``rfft(w) == W(f) / dt`` for the wavelet
    sampled at ``dt``, so a synthetic spectrum is ``S(f) * W(f) / dt`` on the
    rfft grid of the trace.

    Parameters
    ----------
    freq_hz : float or ndarray
        Frequency, Hz (negative values are fine; W is even).
    f_dom_hz : float
        Dominant frequency, Hz (> 0).

    Returns
    -------
    float or ndarray
        Spectrum amplitude, s.
    """
    if not np.isfinite(f_dom_hz) or f_dom_hz <= 0.0:
        raise ValueError(f"f_dom_hz must be finite and > 0, got {f_dom_hz}")
    f = np.asarray(freq_hz, dtype=np.float64)
    spectrum = 2.0 / (np.sqrt(np.pi) * f_dom_hz**3) * f**2 * np.exp(-((f / f_dom_hz) ** 2))
    if f.ndim == 0:
        return spectrum.reshape(-1)[0].item()
    return spectrum
