"""Neidell-Taner semblance analysis on offset gathers (M5).

For every trial pair ``(T0, V_stk)`` the gather is sampled along the hyperbola
``t(x) = sqrt(T0^2 + (x / V_stk)^2)`` over a centered gate of duration
``gate_s`` (linear interpolation between record samples) and scored with the
Neidell-Taner coherency

    S(T0, V) = sum_j ( sum_i d_i(t_j(x_i)) )^2
               / ( N * sum_i sum_j d_i(t_j(x_i))^2 )

with ``i`` over traces, ``j`` over gate samples and ``N`` the number of traces
that contribute: by Cauchy-Schwarz ``0 <= S <= 1``, ``S = 1`` for identical
traces and ``S ~ 1/N`` for incoherent noise.  Being amplitude-normalized, weak
coherent multiples score as highly as strong primaries -- exactly what
multiple analysis needs.  Two exclusions keep cells meaningful:

- traces whose gate leaves the record are excluded pair-wise (a moveout
  steeper than the record can only be scored on the offsets that contain
  it);
- with ``p_mute_s_m`` (the plane-wave synthesis mute of the gather,
  :func:`mm1d.gathers.critical_mute_slowness`), traces whose hyperbola-local
  slowness ``p = dt/dx = x / (V^2 t(x))`` reaches the mute are excluded: no
  plane wave beyond the mute entered the synthesis, so such offsets cannot
  host this event -- only aperture-edge junk.  The criterion is
  event-agnostic (a pure function of the trial hyperbola) and adapts the
  usable aperture per cell: shallow/fast events scan their near offsets
  only, deep slow events the full spread.

Cells where fewer than ``min_traces_fraction`` of the traces contribute are
set to 0 (unconstrained).

**Pick well-posedness on deterministic synthetics**: S is scale-free per
cell, so on noise-free gathers every gate placement inside an event's
wavelet extent ties near ``S ~ 1`` (a vertical ridge wider than the wavelet)
and the panel argmax is ill-posed.  Field data always carries incoherent
noise, which can break ties toward the alignment cell. Raw noiseless data
are the validation default; an optional, explicitly labelled caller-side
noise experiment must not be mistaken for artifact removal. Fold/support
are available from :func:`semblance_with_fold`.

The same panel function serves all three wavefield variants, so ridge
patterns compare directly between them.  Scan grids are auto-scaled:
:func:`default_t0_grid` spans the record length; :func:`default_velocity_grid`
spans ``[low_factor * min(vp), high_factor * max Dix V_rms]`` -- the model
enters only for axis sizing (the panel itself is pure data processing of the
gather; the Dix values it is compared against come from
:func:`dix_rms_velocities`, the same helper used for overlays and tests).

Pick-bias notes (stated on the script figures):

- Dix velocities are *small-spread* stacking velocities.  At large
  offset/depth ratios the true Snell-parameter moveout is non-hyperbolic and
  drags the semblance maximum above Dix (e.g. +25-35 m/s at the full M4
  aperture for T2/T3 primaries); scans therefore use a moderate aperture --
  offsets up to ``~1.1 km/s`` times the deepest primary T0 -- where the
  hyperbola fits the parametric moveout to within a few ms and picks stay
  within about one velocity cell of Dix for T1-T3.
- The scan respects the synthesis aperture via ``p_mute_s_m`` (pass
  :func:`scan_slowness_bound`: the slowness where the synthesis taper
  starts): trial hyperbolas are scored only on offsets whose local slowness
  is inside the *clean* aperture, where M4 documented event fidelity
  (beyond the taper, event tails are attenuated and phase-shifted ~10 ms).
- Resolution floor: at the taper-clean aperture the *shallowest* interface
  of each model has ``x_max / T0 ~ 0.7`` -- its stacking velocity is only
  resolved to ~+/-100 m/s at 20 Hz (a uniform gate shift inside the broad
  wavelet is a near-symmetry of S).  Deep interfaces, whose events span the
  full aperture, pin to Dix within about one velocity cell.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mm1d.gathers import DEFAULT_APERTURE_TAPER, critical_mute_slowness, slowness_from_angle
from mm1d.model import LayeredModel

__all__ = [
    "SemblanceResult",
    "default_t0_grid",
    "default_velocity_grid",
    "dix_rms_velocities",
    "scan_slowness_bound",
    "semblance_panel",
    "semblance_with_fold",
]

_CHUNK: int = 128
"""Number of T0 cells evaluated per vectorized chunk (memory/speed balance)."""


def dix_rms_velocities(model: LayeredModel) -> NDArray[np.float64]:
    """Dix RMS (stacking) velocity of every interface primary, m/s.

    ``V_rms_k = sqrt( sum_{j<=k} twt_j v_j^2 / sum_{j<=k} twt_j )`` with
    two-way layer times ``twt_j = 2 h_j / v_j`` -- the small-spread stacking
    velocity of the primary from interface ``k``.  Used for scan-grid sizing
    and reference overlays; M6's event catalog generalizes this to arbitrary
    bounce paths.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).

    Returns
    -------
    ndarray
        RMS velocity per interface, m/s, shape ``(n_interfaces,)``.
    """
    twt = model.layer_twt_s
    return np.sqrt(np.cumsum(twt * model.vp_m_s[:-1] ** 2) / np.cumsum(twt))


def default_t0_grid(record_s: float, n_t0: int = 301) -> NDArray[np.float64]:
    """Uniform zero-offset scan grid covering a record, s.

    ``n_t0`` points from 0 with step ``record_s / n_t0`` (endpoint excluded:
    the last record sample cannot center a gate).  Step is ~10 ms for the
    canonical records at the default size -- well below the wavelet period,
    so event picks land within half a cell of their true T0.

    Parameters
    ----------
    record_s : float
        Record length to scan, s (> 0).
    n_t0 : int, optional
        Number of grid points (>= 2, default 301).

    Returns
    -------
    ndarray
        Strictly increasing T0 grid, s, shape ``(n_t0,)``.
    """
    if not np.isfinite(record_s) or record_s <= 0.0:
        raise ValueError(f"record_s must be finite and > 0, got {record_s}")
    if n_t0 < 2:
        raise ValueError(f"n_t0 must be >= 2, got {n_t0}")
    step = record_s / n_t0
    return np.arange(n_t0, dtype=np.float64) * step


def default_velocity_grid(
    model: LayeredModel,
    n_v: int = 81,
    *,
    low_factor: float = 0.8,
    high_factor: float = 1.25,
) -> NDArray[np.float64]:
    """Uniform stacking-velocity scan grid auto-scaled to a model, m/s.

    Bounds: ``low_factor * min(vp)`` (surface-related multiples are pinned at
    the layer velocities of their generating paths, down to the slowest layer
    -- the water bottom for T2) to ``high_factor * max Dix V_rms`` (the
    primary trend plus headroom; interbed/peg-leg ridges of normally graded
    sections stay below it).  The model enters only for axis sizing -- the
    semblance panel itself is computed from the gather alone.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model (last layer is the half-space).
    n_v : int, optional
        Number of grid points (>= 2, default 81; ~17-21 m/s spacing for
        T1-T3, about the non-hyperbolic pick bias at moderate apertures).
    low_factor : float, optional
        Fraction of the minimum layer velocity for the slow bound (0, 1].
    high_factor : float, optional
        Fraction of the maximum Dix RMS velocity for the fast bound (>= 1).

    Returns
    -------
    ndarray
        Strictly increasing velocity grid, m/s, shape ``(n_v,)``.
    """
    if n_v < 2:
        raise ValueError(f"n_v must be >= 2, got {n_v}")
    if not 0.0 < low_factor <= 1.0:
        raise ValueError(f"low_factor must be within (0, 1], got {low_factor}")
    if high_factor < 1.0:
        raise ValueError(f"high_factor must be >= 1, got {high_factor}")
    v_low = low_factor * float(np.min(model.vp_m_s))
    v_high = high_factor * float(np.max(dix_rms_velocities(model)))
    if not v_low < v_high:
        raise ValueError(
            f"degenerate velocity range [{v_low}, {v_high}] m/s "
            f"(low_factor * min(vp) must stay below high_factor * max Dix V_rms)"
        )
    return np.linspace(v_low, v_high, n_v)


def scan_slowness_bound(
    model: LayeredModel,
    angles_deg: NDArray[np.float64],
    taper_fraction: float = DEFAULT_APERTURE_TAPER,
) -> float:
    """Slowness bound of the *taper-clean* scan aperture, s/m.

    ``(1 - taper_fraction) * p_end`` with ``p_end = min(critical mute,
    slowness of the largest synthesis angle)``: beyond the synthesis taper
    (the last ``taper_fraction`` of the aperture, or the post-critical mute
    where it comes first) the M4 synthesis documents degraded event fidelity
    -- attenuated, phase-shifted Fresnel tails -- so semblance scans should
    not sum trial hyperbolas past it.  Pass the result as ``p_mute_s_m`` of
    :func:`semblance_panel`.

    Parameters
    ----------
    model : LayeredModel
        Blocked layered model the gather was synthesized from.
    angles_deg : ndarray
        Synthesis aperture angles (the grid given to
        :func:`mm1d.gathers.offset_gather`), degrees in the surface layer.
    taper_fraction : float, optional
        Synthesis aperture taper fraction (default
        :data:`mm1d.gathers.DEFAULT_APERTURE_TAPER`).

    Returns
    -------
    float
        Scan slowness bound, s/m (> 0).
    """
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must be a non-empty finite angle grid")
    if not 0.0 <= taper_fraction < 1.0:
        raise ValueError(f"taper_fraction must be within [0, 1), got {taper_fraction}")
    p_edge = float(np.max(np.atleast_1d(np.asarray(slowness_from_angle(angles, model.vp_m_s[0])))))
    p_mute = critical_mute_slowness(model)
    p_end = min(p_mute, p_edge) if p_mute is not None else p_edge
    bound = (1.0 - taper_fraction) * p_end
    if bound <= 0.0:
        raise ValueError("degenerate scan bound: zero clean aperture")
    return bound


def _validated_scan_grid(grid: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Validate a 1-D finite strictly increasing scan grid."""
    values = np.atleast_1d(np.asarray(grid, dtype=np.float64))
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {values.shape}")
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise ValueError(f"{name} contains non-finite value(s) at index {bad[:5].tolist()}")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing (a scan grid)")
    return values


@dataclass(frozen=True)
class SemblanceResult:
    """Coherence (0..1) and contributing trace count on the same (V,T0) grid."""

    panel: NDArray[np.float64]
    fold: NDArray[np.int64]


def semblance_panel(
    gather: NDArray[np.float64],
    offsets_m: NDArray[np.float64],
    dt_s: float,
    t0_grid_s: NDArray[np.float64],
    v_grid_m_s: NDArray[np.float64],
    gate_s: float,
    *,
    min_traces_fraction: float = 0.2,
    p_mute_s_m: float | None = None,
    valid_samples: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Return coherence only; arguments/units as :func:`semblance_with_fold`.

    Existing ndarray API retained. Use semblance_with_fold for support-aware
    interpretation; valid_samples excludes gates touching missing ray events.
    """
    return semblance_with_fold(
        gather,
        offsets_m,
        dt_s,
        t0_grid_s,
        v_grid_m_s,
        gate_s,
        min_traces_fraction=min_traces_fraction,
        p_mute_s_m=p_mute_s_m,
        valid_samples=valid_samples,
    ).panel


def semblance_with_fold(
    gather: NDArray[np.float64],
    offsets_m: NDArray[np.float64],
    dt_s: float,
    t0_grid_s: NDArray[np.float64],
    v_grid_m_s: NDArray[np.float64],
    gate_s: float,
    *,
    min_traces_fraction: float = 0.2,
    p_mute_s_m: float | None = None,
    valid_samples: NDArray[np.bool_] | None = None,
) -> SemblanceResult:
    """Neidell-Taner semblance panel of an offset gather over (T0, V_stk).

    Sliding-window semblance on hyperbolic moveout (see the module docstring
    for the formula, exclusions, and the noise-floor recommendation for
    noise-free synthetics).  The gate is centered on the hyperbola
    ``t(x) = sqrt(T0^2 + (x/V)^2)`` and sampled by linear interpolation;
    traces whose gate leaves the record ``[0, (n_samples - 1) dt)`` or whose
    hyperbola-local slowness reaches ``p_mute_s_m`` are excluded for that
    cell; cells with fewer contributing traces than
    ``max(2, ceil(min_traces_fraction * n_offsets))`` are set to 0.

    Parameters
    ----------
    gather : ndarray
        Offset gather, shape ``(n_offsets, n_samples)`` (e.g. from
        :func:`mm1d.gathers.offset_gather`; rows must be finite).
    offsets_m : ndarray
        Source-receiver offsets of the traces, m (1-D, finite, strictly
        increasing, same length as ``gather`` rows).
    dt_s : float
        Sample interval, s (> 0).
    t0_grid_s : ndarray
        Zero-offset (intercept) scan times, s (1-D, finite, strictly
        increasing; non-uniform grids are evaluated correctly).
    v_grid_m_s : ndarray
        Stacking-velocity scan values, m/s (1-D, finite, strictly
        increasing).
    gate_s : float
        Duration of the summation gate centered on the hyperbola, s (> 0;
        rounded to the nearest whole number of samples, at least one).
    min_traces_fraction : float, optional
        Minimum fraction of traces with a fully in-record gate for a cell to
        be constrained (default 0.2 -- small enough for shallow events whose
        physical aperture is a fraction of the spread; within (0, 1]).
    p_mute_s_m : float, optional
        Scan slowness bound of the gather, s/m (from
        :func:`scan_slowness_bound`); traces whose trial hyperbola has local
        slowness at or beyond it are excluded (default None: no slowness
        bound).

    valid_samples : ndarray of bool, optional
        Shape matching gather; gates touching invalid samples (including
        interpolation neighbours) are excluded. Use ray backend validity,
        not a slant-stack taper restriction, for diagnostic gathers.

    Returns
    -------
    SemblanceResult
        panel and fold arrays of shape (n_v, n_t0). Panel is 0 where
        unconstrained; fold reports actual support even below minimum fold.
    """
    traces = np.asarray(gather, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError(f"gather must be 2-D (n_offsets, n_samples), got shape {traces.shape}")
    n_x, n_t = traces.shape
    if n_x < 2:
        raise ValueError(f"gather must contain at least 2 offset traces, got {n_x}")
    if n_t < 2:
        raise ValueError(f"gather must contain at least 2 time samples, got {n_t}")
    if not np.all(np.isfinite(traces)):
        raise ValueError("gather contains non-finite value(s)")
    validity = (
        np.ones(traces.shape, dtype=bool) if valid_samples is None else np.asarray(valid_samples)
    )
    if validity.shape != traces.shape or validity.dtype != np.bool_:
        raise ValueError("valid_samples must be a bool array matching gather shape")
    offsets = _validated_scan_grid(offsets_m, "offsets_m")
    if offsets.size != n_x:
        raise ValueError(f"offsets_m has {offsets.size} entries but gather has {n_x} rows")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and > 0, got {dt_s}")
    if not np.isfinite(gate_s) or gate_s <= 0.0:
        raise ValueError(f"gate_s must be finite and > 0, got {gate_s}")
    if not 0.0 < min_traces_fraction <= 1.0:
        raise ValueError(f"min_traces_fraction must be within (0, 1], got {min_traces_fraction}")
    if p_mute_s_m is not None and (not np.isfinite(p_mute_s_m) or p_mute_s_m <= 0.0):
        raise ValueError(f"p_mute_s_m must be finite and > 0, got {p_mute_s_m}")
    t0_grid = _validated_scan_grid(t0_grid_s, "t0_grid_s")
    v_grid = _validated_scan_grid(v_grid_m_s, "v_grid_m_s")

    n_gate = max(1, round(gate_s / dt_s))
    gate_times = (np.arange(n_gate, dtype=np.float64) - (n_gate - 1) / 2.0) * dt_s
    x_index = np.arange(n_x)[np.newaxis, :, np.newaxis]
    min_traces = max(2, int(np.ceil(min_traces_fraction * n_x)))
    p_cut = p_mute_s_m * (1.0 - 1.0e-9) if p_mute_s_m is not None else None

    panel = np.zeros((v_grid.size, t0_grid.size), dtype=np.float64)
    fold = np.zeros(panel.shape, dtype=np.int64)
    t0_squared = t0_grid**2
    for i_v, velocity in enumerate(v_grid):
        # hyperbola pick times per (T0, offset): shape (n_t0, n_x)
        t_pick = np.sqrt(t0_squared[:, np.newaxis] + (offsets[np.newaxis, :] / velocity) ** 2)
        if p_cut is not None:
            # hyperbola-local slowness p = dt/dx = x / (V^2 t); at t = 0
            # (T0 = 0, x = 0) define p = 0 (normal incidence, always valid)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_local = np.where(
                    t_pick > 0.0, np.abs(offsets[np.newaxis, :]) / (velocity**2 * t_pick), 0.0
                )
            slow = p_local >= p_cut  # (n_t0, n_x): offset cannot host this event
        else:
            slow = np.zeros(t_pick.shape, dtype=bool)
        for start in range(0, t0_grid.size, _CHUNK):
            window = slice(start, min(start + _CHUNK, t0_grid.size))
            times = t_pick[window][:, :, np.newaxis] + gate_times  # (m, n_x, n_gate)
            position = times / dt_s
            i0 = np.floor(position).astype(np.intp)
            frac = position - i0
            in_record = (
                (times >= 0.0) & (i0 >= 0) & (i0 + 1 <= n_t - 1)
            )  # linear interpolation needs both neighbours
            i0_clipped = np.clip(i0, 0, n_t - 2)
            amplitudes = (
                traces[x_index, i0_clipped] * (1.0 - frac) + traces[x_index, i0_clipped + 1] * frac
            )
            supported = validity[x_index, i0_clipped] & validity[x_index, i0_clipped + 1]
            trace_ok = (in_record & supported).all(axis=-1) & ~slow[window]
            amplitudes = np.where(trace_ok[:, :, np.newaxis], amplitudes, 0.0)
            n_valid = trace_ok.sum(axis=-1)
            fold[i_v, window] = n_valid
            numerator = (amplitudes.sum(axis=1) ** 2).sum(axis=-1)
            denominator = n_valid * (amplitudes**2).sum(axis=(1, 2))
            good = (n_valid >= min_traces) & (denominator > 0.0)
            panel[i_v, window] = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=good
            )
    return SemblanceResult(panel, fold)
