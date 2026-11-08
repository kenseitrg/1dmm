"""Zero-offset, angle gathers, and angle-to-offset conversion.

Angle gathers (M3) assemble the plane-wave omega-p engine
(:func:`mm1d.reflectivity.stack_response`) over a grid of incidence angles
``theta`` in the surface layer: ``p = sin(theta) / vp[0]`` (angles in degrees
at the public API).  Events appear at their *intercept* times
``tau(p) = sum of q_i h_i over the path legs`` (vertical slowness ``q_i =``
``sqrt(1/v_i^2 - p^2)`` per leg) -- e.g. the primary from interface ``k`` at
``2 sum_{j<=k} q_j h_j``, which decreases with angle; the complementary
offset information (``d tau/d p = -x``) is recovered by Snell ray tracing in
M4.  Amplitudes are plane-wave pressures: no geometric spreading and no
obliquity factor (stated on every angle-gather figure).
"""

import numpy as np
from numpy.typing import NDArray

from mm1d.model import LayeredModel
from mm1d.reflectivity import (
    WavefieldVariant,
    _validate_sampling,
    stack_response,
)
from mm1d.units import deg_to_rad
from mm1d.wavelet import ricker_spectrum

__all__ = [
    "angle_gather",
    "angle_impulse_gather",
    "critical_mute_slowness",
    "primary_intercept_times",
    "slowness_from_angle",
]


def critical_mute_slowness(model: LayeredModel) -> float | None:
    """Slowness beyond which angle-gather traces are muted (zeroed), s/m.

    ``p_mute = 1 / max(vp of all layers below the surface)`` = the smallest
    critical slowness among the interfaces (PLAN.md numerical default:
    "evanescent ``p > 1/v_max`` zeroed").  Beyond it at least one interface is
    post-critical, the stack becomes a lossless total reflector (``|U0| = 1``
    exactly), and the free-surface multiple series ``U0 / (1 - R_s U0)`` of
    the ``full`` variant resonates (unbounded plane-wave cavity; ``min |1+U0|``
    ~ 1e-4 on realistic frequency grids).  The head-wave/tunnelling physics
    there is outside v1 scope, so all variants mute the same slowness range
    for panel consistency.  None when no layer below the surface is faster
    than the surface layer (no interface has a critical angle).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model.

    Returns
    -------
    float or None
        Mute slowness, s/m (at the critical angle of the fastest deeper layer),
        or None if no critical angle exists.
    """
    v_max = float(np.max(model.vp_m_s[1:]))
    if v_max <= float(model.vp_m_s[0]):
        return None
    return 1.0 / v_max


def slowness_from_angle(
    angles_deg: float | NDArray[np.float64], vp_m_s: float
) -> float | NDArray[np.float64]:
    """Horizontal slowness of a plane wave at given incidence angle.

    ``p = sin(theta) / vp`` (Snell parameter, constant along the ray path
    through all layers).

    Parameters
    ----------
    angles_deg : float or ndarray
        Incidence angle in the layer of velocity ``vp_m_s``, degrees.
    vp_m_s : float
        P-wave velocity of that layer, m/s.

    Returns
    -------
    float or ndarray
        Horizontal slowness, s/m.
    """
    if not np.isfinite(vp_m_s) or vp_m_s <= 0.0:
        raise ValueError(f"vp_m_s must be finite and > 0, got {vp_m_s}")
    return np.sin(deg_to_rad(angles_deg)) / vp_m_s


def _validated_angles(angles_deg: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate and normalize an angle grid (degrees, 1-D, within [0, 90))."""
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
    if angles.ndim != 1:
        raise ValueError(f"angles_deg must be 1-D, got shape {angles.shape}")
    if angles.size == 0:
        raise ValueError("angles_deg must contain at least one angle")
    bad = np.flatnonzero(~np.isfinite(angles))
    if bad.size:
        raise ValueError(f"angles_deg contains non-finite value(s) at index {bad[:5].tolist()}")
    bad = np.flatnonzero((angles < 0.0) | (angles >= 90.0))
    if bad.size:
        raise ValueError(
            f"angles_deg must be within [0, 90) degrees; violation at {bad[:5].tolist()} "
            f"(at 90 deg the incident wave is grazing/evanescent in the surface layer)"
        )
    return angles


def angle_impulse_gather(
    model: LayeredModel,
    variant: WavefieldVariant,
    angles_deg: NDArray[np.float64],
    dt_s: float,
    n_samples: int,
    *,
    n_fft: int | None = None,
    mute_postcritical: bool = True,
) -> NDArray[np.float64]:
    """Angle-domain impulse-response gather for one wavefield variant.

    One plane-wave impulse-response trace per incidence angle (in the surface
    layer); see the module docstring for the intercept-time/angle behaviour.
    Sampling semantics are as :func:`mm1d.reflectivity.impulse_response`
    (``n_fft`` extends the periodic FFT record so late multiples wrap beyond
    the displayed window).  With ``mute_postcritical`` (default), traces at
    slowness beyond :func:`critical_mute_slowness` are zeroed (resonant
    free-surface cavity / head-wave regime, see there).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    variant : WavefieldVariant
        Which multiple families to include.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, within [0, 90)).
    dt_s : float
        Sample interval, s (> 0).
    n_samples : int
        Number of samples per trace (>= 2).
    n_fft : int, optional
        Length of the underlying FFT record (>= ``n_samples``); see
        :func:`mm1d.reflectivity.impulse_response`.
    mute_postcritical : bool, optional
        Zero traces at ``p >= critical_mute_slowness(model)`` (default True).

    Returns
    -------
    ndarray
        Impulse gather, shape ``(n_angles, n_samples)``.
    """
    n_fft = _validate_sampling(dt_s, n_samples, n_fft)
    angles = _validated_angles(angles_deg)
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    freq = np.fft.rfftfreq(n_fft, d=dt_s)
    response = stack_response(model, variant, freq, p)
    traces = np.fft.irfft(response, n=n_fft, axis=-1)[:, :n_samples]
    return _apply_mute(model, p, traces, mute_postcritical)


def angle_gather(
    model: LayeredModel,
    variant: WavefieldVariant,
    angles_deg: NDArray[np.float64],
    dt_s: float,
    n_samples: int,
    f_dom_hz: float,
    *,
    n_fft: int | None = None,
    mute_postcritical: bool = True,
) -> NDArray[np.float64]:
    """Angle-domain synthetic gather: impulse response convolved with a Ricker wavelet.

    Unit-peak Ricker source (dominant frequency ``f_dom_hz``); the peak
    amplitude of an isolated event equals its plane-wave impulse amplitude.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    variant : WavefieldVariant
        Which multiple families to include.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, within [0, 90)).
    dt_s : float
        Sample interval, s (> 0).
    n_samples : int
        Number of samples per trace (>= 2).
    f_dom_hz : float
        Dominant frequency of the Ricker source, Hz (> 0).
    n_fft : int, optional
        Length of the underlying FFT record (>= ``n_samples``); see
        :func:`mm1d.reflectivity.impulse_response`.
    mute_postcritical : bool, optional
        Zero traces at ``p >= critical_mute_slowness(model)`` (default True).

    Returns
    -------
    ndarray
        Synthetic gather, shape ``(n_angles, n_samples)``.
    """
    n_fft = _validate_sampling(dt_s, n_samples, n_fft)
    angles = _validated_angles(angles_deg)
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    freq = np.fft.rfftfreq(n_fft, d=dt_s)
    response = stack_response(model, variant, freq, p)
    spectrum = response * np.asarray(ricker_spectrum(freq, f_dom_hz))[np.newaxis, :] / dt_s
    traces = np.fft.irfft(spectrum, n=n_fft, axis=-1)[:, :n_samples]
    return _apply_mute(model, p, traces, mute_postcritical)


def _apply_mute(
    model: LayeredModel,
    p: NDArray[np.float64],
    traces: NDArray[np.float64],
    mute_postcritical: bool,
) -> NDArray[np.float64]:
    """Zero traces beyond the critical mute slowness (see :func:`critical_mute_slowness`)."""
    if not mute_postcritical:
        return traces
    p_mute = critical_mute_slowness(model)
    if p_mute is None:
        return traces
    traces = traces.copy()
    # tiny relative tolerance: at exactly the critical slowness the reflection
    # coefficient of the binding interface is already 1 (resonance), and the
    # grid value may sit a rounding error on either side of it
    traces[p >= p_mute * (1.0 - 1.0e-9), :] = 0.0
    return traces


def primary_intercept_times(
    model: LayeredModel, angles_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Intercept times of the primary events vs incidence angle, s.

    ``tau_k(p) = 2 * sum_{j<=k} h_j q_j`` with ``q_j = sqrt(1/v_j^2 - p^2)``:
    the zero-offset two-way time of interface ``k`` at slowness ``p`` (the
    plane-ray prediction the angle-gather events must land on).  Angles are
    incidence in the surface layer; deeper layers refract by Snell's law
    (``sin(theta_j) = p v_j``).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, within [0, 90)).

    Returns
    -------
    ndarray
        Intercept times, shape ``(n_angles, n_interfaces)``; ``NaN`` where a
        leg down to the interface is evanescent (``p > 1/v_j`` -- total
        reflection/tunnelling regime, no real arrival time).
    """
    angles = _validated_angles(angles_deg)
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    vp = model.vp_m_s[:-1, np.newaxis]
    q_squared = 1.0 / vp**2 - p[np.newaxis, :] ** 2
    propagating = q_squared >= 0.0
    q = np.where(propagating, np.sqrt(np.where(propagating, q_squared, 0.0)), np.nan)
    one_way = model.h_m[:, np.newaxis] * q
    return 2.0 * np.cumsum(one_way, axis=0).T
