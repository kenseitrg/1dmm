"""Zero-offset, angle gathers, and angle-to-offset conversion.

Angle gathers (M3) assemble the plane-wave omega-p engine
(:func:`mm1d.reflectivity.stack_response`) over a grid of incidence angles
``theta`` in the surface layer: ``p = sin(theta) / vp[0]`` (angles in degrees
at the public API).  Events appear at their *intercept* times
``tau(p) = sum of q_i h_i over the path legs`` (vertical slowness ``q_i =``
``sqrt(1/v_i^2 - p^2)`` per leg) -- e.g. the primary from interface ``k`` at
``2 sum_{j<=k} q_j h_j``, which decreases with angle; the complementary
offset information (``d tau/d p = -x``) is recovered in M4.  Amplitudes are
plane-wave pressures: no geometric spreading and no obliquity factor (stated
on every angle-gather figure).

Offset gathers and ray tracing (M4) come in two independent layers:

- **Snell ray tracing** (:func:`path_offset_time`, :func:`primary_offset_times`):
  an event is a *bounce path* through the layer stack (sequence of turning
  points: the free surface or an interface); with the ray parameter ``p``
  constant, each leg contributes ``x += h_i tan(theta_i)`` and
  ``T += h_i / (v_i cos(theta_i))``, so the path lands at its theoretical
  moveout curve ``(x(p), T(p))``.  This is event-aware bookkeeping (used for
  overlays, acceptance anchors, and the M6 event catalog).
- **Offset-gather synthesis** (:func:`offset_gather`,
  :func:`offset_impulse_gather`): the *event-agnostic* transform -- the exact
  inverse of the plane-wave (tau-p) decomposition, an ``|omega|``-filtered
  (rho-filter) inverse slant stack evaluated in the frequency domain.  Every
  event contained in the engine's tau-p response (primaries, surface and
  interbed multiples, peg-legs) automatically lands on its own ``(x, T)``
  curve; no event catalog enters the transform, so semblance panels and
  catalog predictions (M5/M6) remain independent cross-checks of each other.
  :func:`slant_stack` is the matching forward transform (offset gather ->
  tau-p) for round-trip validation.

Synthesis numerics (artifact control): the slowness aperture carries a Tukey
(cos^2) taper over its last ``taper_fraction`` (default 15 %, parameter of
the gather functions) and is refined internally to the anti-aliasing limit
``delta_p <= 1 / (4 f_max x_max)`` for the requested offsets (cap 16x;
``f_max`` = 3x the Ricker dominant frequency, or Nyquist for impulse
gathers).  Literal FFT wrap-around is negligible at the standard 8x record
extension (measured < 3e-5 of peak); the visible artifacts of naive
synthesis are the acausal radiation of the hard aperture truncation (worst
just below the post-critical mute, where ``|U|`` grows toward resonance)
and far-offset quadrature aliasing -- both suppressed to ~1 % of peak by
taper + refinement.  The residual near-offset pre-event ramp (a few % of
peak, deepest events worst) is the events' own non-stationary aperture
content -- the finite-aperture price of amplitude-preserving synthesis
(removing it entirely would require the event catalog, i.e. deposition).

Amplitude convention of the offset gathers (stated on every figure):
2-D line-source plane-wave synthesis.  By stationary phase an event's peak
amplitude at the offset ``x(p*)`` of its stationary ray ``p*`` scales with the
plane-wave pressure amplitude ``A(p*)`` divided by ``sqrt(2 pi |dx/dp|(p*))``
(focusing); the ``|omega|`` rho-filter reshapes the Ricker wavelet by
``sqrt(omega)``.  There is no 3-D geometric spreading.  Event *times* are
exact (group delay is unaffected by the real filter), and amplitudes are
continuous through ``x = 0``.  Stationary rays within roughly a Fresnel
zone of the aperture taper (``p*`` near ``p_max`` or the mute) lose
amplitude gracefully (the taper attenuates them); event *times* stay exact
throughout.
"""

from collections.abc import Sequence
from itertools import pairwise

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
    "DEFAULT_APERTURE_TAPER",
    "angle_gather",
    "angle_impulse_gather",
    "critical_mute_slowness",
    "default_offset_grid",
    "layer_angles",
    "offset_gather",
    "offset_impulse_gather",
    "path_offset_time",
    "primary_intercept_times",
    "primary_offset_times",
    "slant_stack",
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


# ---------------------------------------------------------------------------
# Snell ray tracing (M4): theoretical moveout curves x(p), T(p) of bounce paths
# ---------------------------------------------------------------------------


SURFACE: int = -1
"""Turning-point sentinel for the free surface in bounce paths (see :func:`path_offset_time`)."""


def layer_angles(model: LayeredModel, angles_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Refracted ray angle in every layer vs incidence angle in the surface layer.

    Snell's law with constant ray parameter ``p = sin(theta)/vp[0]``:
    ``sin(theta_j) = p * vp_j``, so rays bend toward the normal in slow layers
    and away in fast ones.  Angles are in degrees at the public API.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, within [0, 90)).

    Returns
    -------
    ndarray
        Layer angles, shape ``(n_layers, n_angles)``, degrees; ``NaN`` where
        the ray is evanescent in that layer (``p >= 1/vp_j``, i.e. beyond the
        layer's critical incidence).
    """
    angles = _validated_angles(angles_deg)
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    sin_theta = p[np.newaxis, :] * model.vp_m_s[:, np.newaxis]
    propagating = np.abs(sin_theta) < 1.0
    cos_theta = np.where(
        propagating, np.sqrt(np.where(propagating, 1.0 - sin_theta**2, 0.0)), np.nan
    )
    return np.degrees(np.arctan2(sin_theta, cos_theta))


def path_offset_time(
    model: LayeredModel,
    turning_points: Sequence[int],
    angles_deg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Theoretical offset and total travel time of one event bounce path vs angle.

    An event path is the sequence of its turning points: ``SURFACE = -1`` for
    the free surface or the interface index ``k`` (base of layer ``k``) for a
    reflection; consecutive points bound one *leg*.  A leg between turning
    points ``a`` and ``b`` crosses the layers ``min(a, b) + 1 .. max(a, b)``
    and, with the ray parameter ``p`` constant (Snell), contributes

    - offset ``x += sum h_j tan(theta_j)`` and
    - time   ``T += sum h_j / (v_j cos(theta_j))``

    over the crossed layers.  Canonical examples: primary of interface ``k``:
    ``[-1, k, -1]``; order-n surface multiple of interface ``k``:
    ``[-1, k] * (n + 1) + [-1]``; an interbed reverberation between
    interfaces ``a < b`` (one bounce at each): ``[-1, b, a, b, -1]``;
    a peg-leg is any concatenation sharing the surface endpoint.  Paths
    conventionally start and end at the surface, but any contiguous bounce
    sequence is accepted (times/offsets are plain leg sums).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model.
    turning_points : sequence of int
        Bounce sequence over ``{-1, 0, ..., n_interfaces - 1}``; at least two
        entries, consecutive entries distinct.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, within [0, 90)).

    Returns
    -------
    tuple of ndarray
        ``(x_s_m, t_s)``, each shape ``(n_angles,)``: source-receiver offset
        in m and total travel time in s.  ``NaN`` at angles where any crossed
        leg is evanescent (``p >= 1/vp_j`` -- grazing or beyond, no real ray).
    """
    points = list(turning_points)
    if len(points) < 2:
        raise ValueError(
            f"turning_points must contain at least 2 entries (a leg needs two "
            f"endpoints), got {len(points)}"
        )
    checked: list[int] = []
    for point in points:
        index = int(point)
        if index != point:
            raise ValueError(f"turning points must be integers, got {point!r}")
        if not SURFACE <= index <= model.n_interfaces - 1:
            raise ValueError(
                f"turning point {index} out of range [{SURFACE}, {model.n_interfaces - 1}] "
                f"(-1 = free surface, k = base of layer k / interface k)"
            )
        checked.append(index)
    for a, b in pairwise(checked):
        if a == b:
            raise ValueError(
                f"consecutive turning points must differ (zero-length leg at {a}); path = {checked}"
            )

    angles = _validated_angles(angles_deg)
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    sin_theta = p[np.newaxis, :] * model.vp_m_s[:-1, np.newaxis]
    propagating = np.abs(sin_theta) < 1.0  # grazing (sin = 1) -> infinite offset
    cos_theta = np.where(
        propagating, np.sqrt(np.where(propagating, 1.0 - sin_theta**2, 0.0)), np.nan
    )
    offset_leg = model.h_m[:, np.newaxis] * sin_theta / cos_theta
    time_leg = model.h_m[:, np.newaxis] / (model.vp_m_s[:-1, np.newaxis] * cos_theta)

    x_total = np.zeros_like(p)
    t_total = np.zeros_like(p)
    for a, b in pairwise(checked):
        lo, hi = sorted((a + 1, b + 1))
        x_total = x_total + offset_leg[lo:hi].sum(axis=0)
        t_total = t_total + time_leg[lo:hi].sum(axis=0)
    return x_total, t_total


def primary_offset_times(
    model: LayeredModel, angles_deg: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Offset and travel time of every primary event vs incidence angle.

    The primary of interface ``k`` is the path ``[-1, k, -1]``:
    ``x_k(p) = 2 sum_{j<=k} h_j tan(theta_j)``,
    ``T_k(p) = 2 sum_{j<=k} h_j / (v_j cos(theta_j))`` -- the parametric
    moveout curve the synthesized offset-gather events must land on, and the
    legs of the M6 stacking-velocity formulas.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, within [0, 90)).

    Returns
    -------
    tuple of ndarray
        ``(x_s_m, t_s)``, each shape ``(n_angles, n_interfaces)``; ``NaN``
        where a down-going leg to that interface is evanescent.
    """
    angles = _validated_angles(angles_deg)
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    sin_theta = p[np.newaxis, :] * model.vp_m_s[:-1, np.newaxis]
    propagating = np.abs(sin_theta) < 1.0
    cos_theta = np.where(
        propagating, np.sqrt(np.where(propagating, 1.0 - sin_theta**2, 0.0)), np.nan
    )
    offset_leg = model.h_m[:, np.newaxis] * sin_theta / cos_theta
    time_leg = model.h_m[:, np.newaxis] / (model.vp_m_s[:-1, np.newaxis] * cos_theta)
    return 2.0 * np.cumsum(offset_leg, axis=0).T, 2.0 * np.cumsum(time_leg, axis=0).T


def default_offset_grid(
    model: LayeredModel,
    angles_deg: NDArray[np.float64],
    n_offsets: int = 61,
) -> NDArray[np.float64]:
    """Uniform offset grid spanning the deepest primary within the usable aperture.

    The far offset is the deepest primary's offset at the largest unmuted
    angle: beyond it no unmuted plane wave has its stationary point, so
    events there would be aperture artifacts.  The near offset is 0 (the
    zero-offset trace lives at ``x = 0``).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, within [0, 90));
        the available acquisition aperture.
    n_offsets : int, optional
        Number of offsets, >= 2 (default 61).

    Returns
    -------
    ndarray
        Uniformly spaced offsets from 0 to ``x_max``, m, shape ``(n_offsets,)``.
    """
    if n_offsets < 2:
        raise ValueError(f"n_offsets must be >= 2, got {n_offsets}")
    angles = _validated_angles(angles_deg)
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    x_primary, _ = primary_offset_times(model, angles)
    usable = np.isfinite(x_primary[:, -1])
    p_mute = critical_mute_slowness(model)
    if p_mute is not None:
        usable &= p < p_mute * (1.0 - 1.0e-9)
    if not np.any(usable):
        raise ValueError(
            "no angle provides a propagating, unmuted ray to the deepest interface; "
            "cannot size an offset grid"
        )
    x_max = float(np.max(x_primary[:, -1][usable]))
    if x_max <= 0.0:
        raise ValueError(f"degenerate offset extent x_max = {x_max} m (all angles zero?)")
    return np.linspace(0.0, x_max, n_offsets)


# ---------------------------------------------------------------------------
# Angle -> offset gather transform (M4): inverse / forward slant stack
# ---------------------------------------------------------------------------


def _trapezoid_weights(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Trapezoidal quadrature weights of a (strictly increasing) 1-D grid."""
    weights = np.empty_like(grid)
    weights[0] = (grid[1] - grid[0]) / 2.0
    weights[1:-1] = (grid[2:] - grid[:-2]) / 2.0
    weights[-1] = (grid[-1] - grid[-2]) / 2.0
    return weights


def _validated_offsets(
    offsets_m: NDArray[np.float64], *, require_grid: bool
) -> NDArray[np.float64]:
    """Validate an offset array (1-D, finite; sorted and >= 2 points if integrated over)."""
    offsets = np.atleast_1d(np.asarray(offsets_m, dtype=np.float64))
    if offsets.ndim != 1:
        raise ValueError(f"offsets_m must be 1-D, got shape {offsets.shape}")
    if offsets.size == 0:
        raise ValueError("offsets_m must contain at least one offset")
    bad = np.flatnonzero(~np.isfinite(offsets))
    if bad.size:
        raise ValueError(f"offsets_m contains non-finite value(s) at index {bad[:5].tolist()}")
    if require_grid:
        if offsets.size < 2:
            raise ValueError(
                "offsets_m must contain at least 2 offsets for the forward slant stack "
                "(it integrates over the offset axis)"
            )
        if np.any(np.diff(offsets) <= 0.0):
            raise ValueError("offsets_m must be strictly increasing for quadrature over x")
    return offsets


DEFAULT_APERTURE_TAPER: float = 0.15
"""Default Tukey (cos^2) taper fraction of the synthesis slowness aperture.

The aperture integral of the inverse slant stack is tapered over its last
``taper_fraction`` (applied below the critical mute when one applies).  A
hard truncation radiates acausally -- worst just below the mute, where the
stack response ``|U|`` grows toward resonance -- at up to ~3 % of the gather
peak; the 15 % Tukey taper suppresses that to ~1 % while leaving interior
stationary rays (and the amplitude anchors) untouched.  Set 0 to recover
the untapered quadrature.  See the module docstring (artifact control).
"""


def _aperture_weights(
    model: LayeredModel,
    angles_deg: NDArray[np.float64],
    mute_postcritical: bool,
    taper_fraction: float = DEFAULT_APERTURE_TAPER,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Slowness grid (from angles), trapezoid weights clipped at the mute, edge-tapered.

    The synthesis integrates over the slowness aperture ``[0, p_end]`` with
    ``p_end = min(last angle, critical mute)``: grid points beyond the cut
    get zero weight and the last point below keeps only its half-interval
    (a plain zeroing would let its weight absorb the truncated neighbour's
    half-step).  A Tukey (cos^2) ramp then tapers the last ``taper_fraction``
    of the surviving range, suppressing the acausal radiation of the hard
    truncation (see :data:`DEFAULT_APERTURE_TAPER`).
    """
    if not 0.0 <= taper_fraction < 1.0:
        raise ValueError(f"taper_fraction must be within [0, 1), got {taper_fraction}")
    angles = _validated_angles(angles_deg)
    if angles.size < 2:
        raise ValueError(
            "angles_deg must contain at least 2 angles for offset-gather synthesis "
            "(it integrates over the slowness aperture)"
        )
    if np.any(np.diff(angles) <= 0.0):
        raise ValueError("angles_deg must be strictly increasing (a monotone p aperture)")
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0]), dtype=np.float64))
    p_mute = critical_mute_slowness(model) if mute_postcritical else None
    if p_mute is None:
        weights = _trapezoid_weights(p)
        p_end = float(p[-1])
    else:
        p_cut = p_mute * (1.0 - 1.0e-9)  # at exactly the critical slowness the stack resonates
        right = np.empty_like(p)
        right[:-1] = np.minimum(p[1:], p_cut)
        right[-1] = p_cut
        left = np.empty_like(p)
        left[0] = p[0]
        left[1:] = p[:-1]
        weights = np.where(p < p_cut, (right - left) / 2.0, 0.0)
        p_end = min(float(p[-1]), p_cut)
    if taper_fraction > 0.0 and p_end > 0.0:
        p_start_taper = (1.0 - taper_fraction) * p_end
        ramp = np.ones_like(p)
        in_taper = p > p_start_taper
        ramp[in_taper] = (
            np.cos(np.pi / 2.0 * (p[in_taper] - p_start_taper) / (taper_fraction * p_end)) ** 2
        )
        weights = weights * ramp
    return p, weights


def _refine_angles(angles: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Densify a strictly increasing angle grid by ``factor`` sub-steps per interval."""
    if factor <= 1:
        return angles
    pieces = [np.linspace(a, b, factor + 1)[:-1] for a, b in pairwise(angles)]
    return np.concatenate([*pieces, angles[-1:]])


def _refine_factor(
    p_s_m: NDArray[np.float64],
    offsets_m: NDArray[np.float64],
    f_max_hz: float,
) -> int:
    """Aperture densification factor for alias-free synthesis quadrature.

    The synthesis kernel oscillates in ``p`` with phase ``omega p x``: the
    trapezoid resolves it while ``delta_p <= 1 / (4 f_max x_max)`` (half the
    quadrature Nyquist, for margin).  Capped at 16x; 1 when there is nothing
    to resolve (zero extent or zero bandwidth).
    """
    if f_max_hz <= 0.0:
        return 1
    x_max = float(np.max(np.abs(offsets_m)))
    if x_max == 0.0:
        return 1
    delta_p = float(np.max(np.diff(p_s_m)))
    needed = 1.0 / (4.0 * f_max_hz * x_max)
    return int(np.clip(np.ceil(delta_p / needed), 1, 16))


def _slant_stack_spectra(
    response_p: NDArray[np.complex128],
    weights: NDArray[np.float64],
    p_s_m: NDArray[np.float64],
    offsets_m: NDArray[np.float64],
    omega_rad_s: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Rho-filtered inverse slant stack in the frequency domain.

    ``D(x, w) = (|w| / 2 pi) * sum over the *mirrored* slowness grid
    {+p_j} U {-p_j} of w_j U(p_j, w) exp(-i w p_j x)``
           ``= (|w| / pi) * sum_j w_j U(p_j, w) cos(w p_j x)``, using that
    the plane-wave response is even in ``p`` (the engine depends on ``p^2``
    only).  The mirrored grid is the symmetric-spread synthesis (plane waves
    leaving in both horizontal directions); its stationary-phase events are
    continuous through ``x = 0`` and the pair with the symmetric forward
    :func:`slant_stack` is an exact inverse, see :func:`offset_impulse_gather`.
    """
    spectra = np.zeros((offsets_m.size, omega_rad_s.size), dtype=np.complex128)
    for j in range(p_s_m.size):
        if weights[j] == 0.0:
            continue
        kernel = np.cos(omega_rad_s[np.newaxis, :] * (p_s_m[j] * offsets_m[:, np.newaxis]))
        spectra += weights[j] * kernel * response_p[j][np.newaxis, :]
    spectra *= omega_rad_s[np.newaxis, :] / np.pi
    return spectra


def offset_impulse_gather(
    model: LayeredModel,
    variant: WavefieldVariant,
    angles_deg: NDArray[np.float64],
    offsets_m: NDArray[np.float64],
    dt_s: float,
    n_samples: int,
    *,
    n_fft: int | None = None,
    mute_postcritical: bool = True,
    taper_fraction: float = DEFAULT_APERTURE_TAPER,
) -> NDArray[np.float64]:
    """Offset-domain impulse-response gather: rho-filtered inverse slant stack.

    Synthesizes the offset gather whose plane-wave (tau-p) transform is the
    engine's response on the given angle grid -- the event-agnostic angle ->
    offset transform of PLAN.md M4.  With the forward slant stack
    ``P[d](p, tau) = integral d(x, tau + p x) dx`` and its adjoint
    ``S[u](x, t) = integral u(p, t - p x) dp``, one has
    ``(S P)(x, w) = (2 pi / |w|) D(x, w)``, so the exact inverse is the
    ``|w|``-filtered (rho-filter) adjoint ``(|w| / 2 pi) S``.  mm1d evaluates
    it on the *mirrored* slowness grid ``{+-p_j}`` (symmetric spread; the
    response is even in ``p``), i.e.
    ``D(x, w) = (|w| / pi) sum_j w_j U(p_j, w) cos(w p_j x)`` with trapezoid
    weights on the (angle-derived, nonuniform) ``p`` grid, clipped at the
    critical mute (:func:`critical_mute_slowness`) and Tukey-tapered over
    the aperture edge (see :data:`DEFAULT_APERTURE_TAPER`); the aperture is
    refined internally to the anti-aliasing limit for ``offsets_m`` (see
    the module docstring, "artifact control").  Every event of ``U`` (any
    multiple family) lands at its true ``(x(p*), T(p*))`` moveout with the
    stationary ray ``p*`` -- no event catalog enters the transform, and
    event amplitudes are continuous through ``x = 0``.

    Note the impulse-domain caveat: the ``|w|`` filter sharpens broadband
    spikes into rho-kernel shapes (oscillatory ``t^{-3/2}`` tails) up to
    Nyquist; the Ricker-band :func:`offset_gather` is the interpretable
    representation (event times are identical).

    Amplitude convention: 2-D line-source plane-wave synthesis (see the
    module docstring) -- event peaks scale as ``A(p*) / sqrt(2 pi |dx/dp|)``
    (in the ``sqrt(omega)``-wavelet measure), no 3-D geometric spreading.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    variant : WavefieldVariant
        Which multiple families to include.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, >= 2 entries,
        strictly increasing -- the synthesis aperture).
    offsets_m : ndarray
        Source-receiver offsets, m (1-D, finite, any order; each offset trace
        is synthesized independently).
    dt_s : float
        Sample interval, s (> 0).
    n_samples : int
        Number of samples per trace (>= 2).
    n_fft : int, optional
        Length of the underlying FFT record (>= ``n_samples``); late events
        delayed by up to ``p_max * x_max`` must fit, so keep the M2/M3-style
        extension (the default scripts use 8x the display window).
    mute_postcritical : bool, optional
        Exclude slowness beyond :func:`critical_mute_slowness` from the
        synthesis integral (default True; matches the angle-gather mute).
    taper_fraction : float, optional
        Tukey taper over the last fraction of the aperture (default
        :data:`DEFAULT_APERTURE_TAPER` = 0.15; 0 disables).  Trades far-edge
        amplitude (stationary rays inside the taper zone) for suppression of
        acausal truncation radiation.

    Returns
    -------
    ndarray
        Impulse gather, shape ``(n_offsets, n_samples)``.
    """
    n_fft = _validate_sampling(dt_s, n_samples, n_fft)
    offsets = _validated_offsets(offsets_m, require_grid=False)
    p, weights = _aperture_weights(model, angles_deg, mute_postcritical, taper_fraction)
    # refine the aperture for alias-free quadrature up to Nyquist (impulse domain)
    factor = _refine_factor(p, offsets, f_max_hz=0.5 / dt_s)
    if factor > 1:
        angles_fine = _refine_angles(_validated_angles(angles_deg), factor)
        p, weights = _aperture_weights(model, angles_fine, mute_postcritical, taper_fraction)
    freq = np.fft.rfftfreq(n_fft, d=dt_s)
    omega = 2.0 * np.pi * freq
    response = stack_response(model, variant, freq, p)
    spectra = _slant_stack_spectra(response, weights, p, offsets, omega)
    return np.fft.irfft(spectra, n=n_fft, axis=-1)[:, :n_samples]


def offset_gather(
    model: LayeredModel,
    variant: WavefieldVariant,
    angles_deg: NDArray[np.float64],
    offsets_m: NDArray[np.float64],
    dt_s: float,
    n_samples: int,
    f_dom_hz: float,
    *,
    n_fft: int | None = None,
    mute_postcritical: bool = True,
    taper_fraction: float = DEFAULT_APERTURE_TAPER,
) -> NDArray[np.float64]:
    """Offset-domain synthetic gather: inverse slant stack of the Ricker-band response.

    As :func:`offset_impulse_gather` with the unit-peak Ricker source
    (dominant frequency ``f_dom_hz``) folded into the synthesis integral;
    the ``|w|`` rho-filter reshapes the wavelet by ``sqrt(w)`` and event peak
    amplitudes equal ``A(p*) / sqrt(2 pi |dx/dp|)`` (2-D line-source
    plane-wave synthesis; times exact, no 3-D spreading).

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    variant : WavefieldVariant
        Which multiple families to include.
    angles_deg : ndarray
        Incidence angles in the surface layer, degrees (1-D, >= 2 entries,
        strictly increasing -- the synthesis aperture).
    offsets_m : ndarray
        Source-receiver offsets, m (1-D, finite; ``0`` gives the zero-offset
        trace location).
    dt_s : float
        Sample interval, s (> 0).
    n_samples : int
        Number of samples per trace (>= 2).
    f_dom_hz : float
        Dominant frequency of the Ricker source, Hz (> 0).
    n_fft : int, optional
        Length of the underlying FFT record (>= ``n_samples``); see
        :func:`offset_impulse_gather`.
    mute_postcritical : bool, optional
        Exclude slowness beyond :func:`critical_mute_slowness` (default True).
    taper_fraction : float, optional
        Tukey taper over the last fraction of the aperture (default
        :data:`DEFAULT_APERTURE_TAPER` = 0.15; 0 disables); see
        :func:`offset_impulse_gather`.

    Returns
    -------
    ndarray
        Synthetic gather, shape ``(n_offsets, n_samples)``.
    """
    n_fft = _validate_sampling(dt_s, n_samples, n_fft)
    offsets = _validated_offsets(offsets_m, require_grid=False)
    p, weights = _aperture_weights(model, angles_deg, mute_postcritical, taper_fraction)
    # refine the aperture for alias-free quadrature over the Ricker band
    factor = _refine_factor(p, offsets, f_max_hz=3.0 * f_dom_hz)
    if factor > 1:
        angles_fine = _refine_angles(_validated_angles(angles_deg), factor)
        p, weights = _aperture_weights(model, angles_fine, mute_postcritical, taper_fraction)
    freq = np.fft.rfftfreq(n_fft, d=dt_s)
    omega = 2.0 * np.pi * freq
    response = stack_response(model, variant, freq, p)
    spectra = _slant_stack_spectra(response, weights, p, offsets, omega)
    spectra = spectra * np.asarray(ricker_spectrum(freq, f_dom_hz))[np.newaxis, :] / dt_s
    return np.fft.irfft(spectra, n=n_fft, axis=-1)[:, :n_samples]


def slant_stack(
    traces: NDArray[np.float64],
    offsets_m: NDArray[np.float64],
    p_s_m: NDArray[np.float64],
    dt_s: float,
    *,
    n_fft: int | None = None,
) -> NDArray[np.float64]:
    """Forward slant stack (tau-p transform) of a one-sided offset gather.

    ``u(p, tau) = integral_0^xmax d(x, tau + p x) dx`` -- the classical
    plane-wave transform of a gather with receivers at ``x >= 0`` only
    (trapezoid on the offset grid, exact ``exp(+i w p x)`` phase shifts in
    the frequency domain).  Eventwise it inverts the rho-filtered synthesis
    of :func:`offset_gather` / :func:`offset_impulse_gather`: an event placed
    at ``(x(p*), T(p*))`` returns to its intercept time ``tau(p*)`` with its
    plane-wave amplitude, up to finite-aperture corrections of order 10-20 %
    (the mirrored ``-p`` background of the exact bi-infinite inverse pair
    does not cancel on one-sided grids; raw traces therefore match in
    correlation and scale, not sample-exactly).  This is the round-trip
    acceptance of M4; keep comparisons away from aperture edges and mutes.

    Parameters
    ----------
    traces : ndarray
        Offset gather, shape ``(n_offsets, n_samples)``.
    offsets_m : ndarray
        Offsets of the traces, m (1-D, strictly increasing, finite, same
        length as ``traces``).
    p_s_m : ndarray
        Horizontal slownesses to transform to, s/m (1-D, finite, any order).
    dt_s : float
        Sample interval, s (> 0).
    n_fft : int, optional
        Length of the underlying FFT record (>= ``n_samples``); shifts are
        periodic in it, so extend beyond the display window.

    Returns
    -------
    ndarray
        Plane-wave traces, shape ``(n_p, n_samples)``.
    """
    gather = np.asarray(traces, dtype=np.float64)
    if gather.ndim != 2:
        raise ValueError(f"traces must be 2-D (n_offsets, n_samples), got shape {gather.shape}")
    n_t = gather.shape[1]
    n_fft = _validate_sampling(dt_s, n_t, n_fft)
    offsets = _validated_offsets(offsets_m, require_grid=True)
    if offsets.size != gather.shape[0]:
        raise ValueError(
            f"offsets_m has {offsets.size} entries but traces has {gather.shape[0]} rows"
        )
    p = np.atleast_1d(np.asarray(p_s_m, dtype=np.float64))
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"p_s_m must be a non-empty 1-D array, got shape {p.shape}")
    if np.any(~np.isfinite(p)):
        raise ValueError("p_s_m contains non-finite value(s)")

    freq = np.fft.rfftfreq(n_fft, d=dt_s)
    omega = 2.0 * np.pi * freq
    spectra_x = np.fft.rfft(gather, n=n_fft, axis=-1)  # (n_x, n_f)
    weights = _trapezoid_weights(offsets)
    spectra_p = np.zeros((p.size, omega.size), dtype=np.complex128)
    for i in range(offsets.size):
        advance = np.exp(1j * omega[np.newaxis, :] * (p[:, np.newaxis] * offsets[i]))
        spectra_p += weights[i] * advance * spectra_x[i][np.newaxis, :]
    return np.fft.irfft(spectra_p, n=n_fft, axis=-1)[:, :n_t]
