"""w-p reflectivity recursion and wavefield variants.

Recursive (compound-matrix style, numerically stable) computation of stack
reflectivity/transmissivity in the frequency-slowness domain per Mueller
(1985), with the three wavefield variants selected by boundary/coefficient
switches: free-surface reflector on/off and interbed reverberation on/off.

This module currently implements the **normal-incidence path** (M2; the
productized ``1dmm_ud.ipynb`` prototype).  The compound recursion is evaluated
per frequency on the exact (blocked) layer stack, so arrival times and
amplitude products are exact rather than sampled-per-cell as in the notebook:

- ``G_i(w)`` is the compound reflection response (up-going/down-going) of the
  sub-stack below interface ``i``, evaluated just above interface ``i``
  inside layer ``i``.  Bottom-up::

      G_{N-1} = R_{N-1}
      G_i     = R_i + Td_i * Tu_i * X / (1 + R_i * X),   X = G_{i+1} e^{-2 i w tau_{i+1}}

  with interface reflection ``R_i``, pressure transmissions ``Td_i, Tu_i``
  (``Td_i * Tu_i = 1 - R_i**2``) and one-way layer time ``tau``.  The
  denominator is the exact geometric sum of the reverberation internal to
  layer ``i + 1`` (reflection from below interface ``i`` is ``-R_i``), i.e.
  all interbed multiples below interface ``i``.
- The surface-recorded response is the up-going wavefield just below the free
  surface (de-ghosted hydrophone-equivalent recording, AGENTS.md): the direct
  sub-stack return ``U_0 = G_0 e^{-2 i w tau_0}`` plus, for ``"full"``, the
  geometric series of surface bounces ``U_0 / (1 - R_s U_0)`` with
  ``R_s = -1`` (pressure-release).

Variant switches (:func:`variant_multiples`) select which denominator/series
terms are kept, matching the notebook's per-cell recursion:

- ``"full"``: surface series + interbed denominators (everything),
- ``"no_surface"``: interbed denominators only (``reflectivity_down[0] = 0``),
- ``"primaries"``: no denominators (all downward interface reflections
  suppressed) -- the single-scattering series with transmission losses.

The same switch semantics carry over to the full w-p engine (M3).
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mm1d.model import LayeredModel
from mm1d.wavelet import ricker_spectrum

__all__ = [
    "ALL_VARIANTS",
    "FREE_SURFACE_REFLECTIVITY",
    "WavefieldVariant",
    "impulse_response",
    "stack_response_normal",
    "variant_multiples",
    "zero_offset_trace",
]

WavefieldVariant = Literal["full", "no_surface", "primaries"]
"""Wavefield variant name (single source of truth for the three variants).

- ``"full"``: primaries + surface-related + interbed multiples
- ``"no_surface"``: primaries + interbed multiples, no free-surface reflections
- ``"primaries"``: primaries only (single scattering)
"""

ALL_VARIANTS: tuple[WavefieldVariant, ...] = ("full", "no_surface", "primaries")
"""Canonical ordering of the three wavefield variants."""

FREE_SURFACE_REFLECTIVITY: float = -1.0
"""Free-surface reflection coefficient (pressure-release surface, AGENTS.md).

An up-going pressure wave reflects back down with amplitude ``-1``.  The
physically equivalent geophone/velocity picture (``+1``) flips every interface
coefficient too; mixing conventions is non-physical.  With ``R_s = -1`` the
surface-multiple train alternates polarity (e.g. ``R * (-R)^n`` for a single
interface) -- the signature validated by the M2 analytic anchors.  Primaries
are unaffected by ``R_s``; only the surface-multiple train depends on it.
"""


def variant_multiples(variant: WavefieldVariant) -> tuple[bool, bool]:
    """Which multiple families a wavefield variant includes.

    Parameters
    ----------
    variant : WavefieldVariant
        One of ``"full"``, ``"no_surface"``, ``"primaries"``.

    Returns
    -------
    tuple of bool
        ``(surface_related, interbed)`` switches.

    Raises
    ------
    ValueError
        If ``variant`` is not one of the three names.
    """
    switches: dict[str, tuple[bool, bool]] = {
        "full": (True, True),
        "no_surface": (False, True),
        "primaries": (False, False),
    }
    try:
        return switches[variant]
    except (KeyError, TypeError) as err:
        options = ", ".join(repr(v) for v in ALL_VARIANTS)
        raise ValueError(
            f"unknown wavefield variant {variant!r}; expected one of {options}"
        ) from err


def stack_response_normal(
    model: LayeredModel,
    variant: WavefieldVariant,
    freq_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Normal-incidence frequency response of the stack for one variant.

    The response maps a downward source impulse emitted at the surface
    (just below the free surface) to the recorded up-going wavefield just
    below the surface, in the frequency domain (``e^{-i w t}`` convention):
    multiplying by ``exp(-i w t0)`` delays by ``t0``.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    variant : WavefieldVariant
        Which multiple families to include.
    freq_hz : ndarray
        Frequencies, Hz; any 1-D grid (negative values allowed, the response
        is Hermitian: ``S(-f) = conj(S(f))``).

    Returns
    -------
    ndarray of complex
        Frequency response ``S(f)`` (dimensionless), same shape as ``freq_hz``.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    omega = 2.0 * np.pi * freq
    reflections = model.reflections_normal
    t_down, t_up = model.transmissions_normal
    one_way_s = model.layer_twt_s / 2.0
    surface, interbed = variant_multiples(variant)

    # compound response of the sub-stack below interface i, seen from layer i
    response = np.full_like(omega, reflections[-1], dtype=np.complex128)
    for i in range(model.n_interfaces - 2, -1, -1):
        # reverberation inside layer i+1: round-trip phase times the deeper
        # compound response; denominator = exact geometric series of internal
        # multiples (reflection from below interface i is -R_i)
        bounce = np.exp(-2j * omega * one_way_s[i + 1]) * response
        denominator = 1.0 + reflections[i] * bounce if interbed else 1.0
        response = reflections[i] + t_down[i] * t_up[i] * bounce / denominator

    # propagate the return through layer 0 to the surface (two-way time 2 tau_0)
    surface_return = np.exp(-2j * omega * one_way_s[0]) * response
    if surface:
        # geometric series of free-surface bounces: U0 / (1 - R_s U0)
        surface_return = surface_return / (1.0 - FREE_SURFACE_REFLECTIVITY * surface_return)
    return surface_return


def _validate_sampling(dt_s: float, n_samples: int, n_fft: int | None) -> int:
    """Validate the sampling parameters and return the FFT length."""
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and > 0, got {dt_s}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if n_fft is None:
        n_fft = n_samples
    elif n_fft < n_samples:
        raise ValueError(f"n_fft must be >= n_samples = {n_samples}, got {n_fft}")
    return n_fft


def impulse_response(
    model: LayeredModel,
    variant: WavefieldVariant,
    dt_s: float,
    n_samples: int,
    *,
    n_fft: int | None = None,
) -> NDArray[np.float64]:
    """Zero-offset impulse response of the stack for one variant.

    The exact frequency response is sampled on the FFT grid implied by
    ``(dt_s, n_fft)`` (default ``n_fft = n_samples``) and inverse-transformed,
    so the result is the periodic extension of the impulse response with
    period ``T_fft = n_fft * dt_s``.  Arrival amplitudes are exact to machine
    precision when arrival times fall on sample points (blocked canonical
    models are built that way); energy from multiples beyond ``T_fft`` wraps
    around into the record -- the wrapped contamination decays roughly
    exponentially with ``T_fft``, so amplitude-critical work passes
    ``n_fft`` several times longer than the window of interest (the
    analytic-anchor tests use ``T_fft`` of 8-32 s).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    variant : WavefieldVariant
        Which multiple families to include.
    dt_s : float
        Sample interval, s (> 0).
    n_samples : int
        Number of samples returned (>= 2); display window
        ``T = n_samples * dt_s``.
    n_fft : int, optional
        Length of the underlying FFT record (>= ``n_samples``); extends the
        periodicity so late multiples wrap beyond the returned window
        instead of into it.

    Returns
    -------
    ndarray
        Impulse response trace, shape ``(n_samples,)``.
    """
    n_fft = _validate_sampling(dt_s, n_samples, n_fft)
    freq = np.fft.rfftfreq(n_fft, d=dt_s)
    response = stack_response_normal(model, variant, freq)
    return np.fft.irfft(response, n=n_fft)[:n_samples]


def zero_offset_trace(
    model: LayeredModel,
    variant: WavefieldVariant,
    dt_s: float,
    n_samples: int,
    f_dom_hz: float,
    *,
    n_fft: int | None = None,
) -> NDArray[np.float64]:
    """Zero-offset synthetic trace: impulse response convolved with a Ricker wavelet.

    The source is a unit-peak Ricker (dominant frequency ``f_dom_hz``) emitted
    downward at the surface; the trace is the up-going wavefield just below
    the surface (see :func:`stack_response_normal`).  The peak amplitude of an
    isolated event equals its impulse amplitude (wavelet peak = 1).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    variant : WavefieldVariant
        Which multiple families to include.
    dt_s : float
        Sample interval, s (> 0).
    n_samples : int
        Number of samples returned (>= 2); display window
        ``T = n_samples * dt_s``.
    f_dom_hz : float
        Dominant frequency of the Ricker source, Hz (> 0).
    n_fft : int, optional
        Length of the underlying FFT record (>= ``n_samples``); see
        :func:`impulse_response`.

    Returns
    -------
    ndarray
        Synthetic trace, shape ``(n_samples,)``.
    """
    n_fft = _validate_sampling(dt_s, n_samples, n_fft)
    freq = np.fft.rfftfreq(n_fft, d=dt_s)
    response = stack_response_normal(model, variant, freq)
    spectrum = response * ricker_spectrum(freq, f_dom_hz) / dt_s
    return np.fft.irfft(spectrum, n=n_fft)[:n_samples]
