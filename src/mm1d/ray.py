"""Diagnostic ray-path gathers: pressure R/T products, unchanged unit Ricker.

Not a point/line-source Green function: no diffraction or geometric spreading.
The independent omega-p engine remains the physical plane-wave cross-check.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from mm1d.events import Event, EventCatalog, enumerate_events, make_event, path_amplitudes
from mm1d.gathers import _validated_offsets, slowness_from_angle
from mm1d.model import LayeredModel
from mm1d.reflectivity import WavefieldVariant, _validate_sampling
from mm1d.wavelet import ricker_wavelet


@dataclass(frozen=True)
class RayGather:
    """Diagnostic synthesis with explicit event support and truncation metadata.

    traces: (offset, time) dimensionless pressure. supported: (event, offset)
    bool; arrival_times_s and slownesses_s_m have that shape too, with NaN for
    unsupported rays. valid_samples: (offset, time) bool, false within the
    declared wavelet window of any unsupported event. Outside that window,
    missing tails are bounded per event by wavelet_tail_tolerance times |A|,
    not by a global gather error. No missing trace is silently treated as data.
    """

    traces: NDArray[np.float64]
    catalog: EventCatalog
    supported: NDArray[np.bool_]
    arrival_times_s: NDArray[np.float64]
    slownesses_s_m: NDArray[np.float64]
    valid_samples: NDArray[np.bool_]
    p_max_s_m: float
    wavelet_margin_s: float


def wavelet_tail_margin(f_dom_hz: float, tolerance: float = 1e-10) -> float:
    """Ricker half-width (s) beyond which absolute unit-wavelet tails <= tolerance.

    f_dom_hz is dominant frequency (Hz); tolerance is a dimensionless per-event
    tail bound, in (0, .1). This is not an amplitude pruning threshold.
    """
    if not np.isfinite(f_dom_hz) or f_dom_hz <= 0:
        raise ValueError("f_dom_hz must be finite and > 0 (Hz)")
    if not np.isfinite(tolerance) or not 0 < tolerance < 0.1:
        raise ValueError("wavelet tail tolerance must be in (0, .1)")
    z = brentq(lambda z: (2 * z * z - 1) * np.exp(-z * z) - tolerance, 2.0, 30.0)
    return float(z / (np.pi * f_dom_hz))


def event_moveout(
    model: LayeredModel,
    event: Event,
    offsets_m: NDArray[np.float64],
    max_angle_deg: float = 40.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Solve Snell moveout for offsets (m), returning p (s/m), T (s), support.

    max_angle_deg is the surface aperture (degrees, [0,90)). Only globally
    subcritical rays are supported, even if a shallow path would propagate
    farther. Geometry outside this support is used ONLY to locate missing-data
    windows; it never licenses synthesis or pressure coefficient evaluation.
    """
    offsets = _validated_offsets(offsets_m, require_grid=False)
    if not np.isfinite(max_angle_deg) or not 0 <= max_angle_deg < 90:
        raise ValueError("max_angle_deg must be in [0, 90) degrees")
    if make_event(model, event.turning_points) != event:
        raise ValueError("event does not match model geometry")
    counts = np.asarray(event.traversal_counts)
    if counts.shape != model.h_m.shape or np.any(counts < 0) or not np.any(counts):
        raise ValueError("invalid traversal counts for model")
    active = counts > 0  # avoid 0 * NaN for untraversed layers
    length = counts[active] * model.h_m[active]
    v = model.vp_m_s[:-1][active]
    p_max = min(
        float(slowness_from_angle(max_angle_deg, model.vp_m_s[0])),
        float(1 / np.max(model.vp_m_s)) * (1 - 1e-6),
    )

    def offset(p: float) -> float:
        q = np.sqrt(1 / v**2 - p * p)
        return float(np.sum(length * p / q))

    def time(p: float) -> float:
        q = np.sqrt(1 / v**2 - p * p)
        return float(np.sum(length / (v * v * q)))

    limit = offset(p_max)
    p_geometry = (1 / float(np.max(v))) * (1 - 1e-12)
    params = np.zeros_like(offsets)
    times = np.zeros_like(offsets)
    support = np.abs(offsets) <= limit
    for i, x in enumerate(np.abs(offsets)):
        if x == 0:
            params[i] = 0.0
        else:
            upper = p_max if support[i] else p_geometry
            if offset(upper) < x:
                params[i], times[i] = np.nan, np.nan
                continue
            params[i] = brentq(lambda p, x=x: offset(p) - x, 0.0, upper, xtol=1e-16, rtol=1e-14)
        times[i] = time(float(params[i]))
    return params, times, support


def ray_offset_gather(
    model: LayeredModel,
    variant: WavefieldVariant,
    offsets_m: NDArray[np.float64],
    dt_s: float,
    n_samples: int,
    f_dom_hz: float,
    *,
    max_angle_deg: float = 40.0,
    max_reflections: int = 32,
    max_paths: int = 100_000,
    wavelet_tail_tolerance: float = 1e-10,
) -> RayGather:
    """Sum bounded paths at exact fractional ray times with R/T-only amplitudes.

    Parameters
    ----------
    model : LayeredModel
        Acoustic model (SI units).
    variant : WavefieldVariant
        Shared full/no_surface/primaries switches.
    offsets_m : ndarray
        Finite offsets, m, any order; negative offsets are symmetric.
    dt_s : float
        Sample interval, s (>0).
    n_samples : int
        Record sample count (>=2).
    f_dom_hz : float
        Unit-peak Ricker dominant frequency, Hz (>0).
    max_angle_deg : float, optional
        Surface ray aperture, degrees (default 40); no slant-stack taper.
    max_reflections, max_paths : int, optional
        Enumeration bounds; see enumerate_events. Defaults 32 / 100000.
    wavelet_tail_tolerance : float, optional
        Per-event absolute unit-wavelet tail tolerance, default 1e-10.
        Catalog includes centers through record end + this tail margin.

    Returns
    -------
    RayGather
        Traces, event-specific support, sample validity, and reached bounds.
        Unsupported events are not clipped/deposited at the aperture boundary.
        Reflection-truncated catalogs are returned explicitly marked; callers
        must establish order convergence before interpreting weak energy.
    """
    _validate_sampling(dt_s, n_samples, None)
    offsets = _validated_offsets(offsets_m, require_grid=False)
    margin = wavelet_tail_margin(f_dom_hz, wavelet_tail_tolerance)
    if not np.isfinite(max_angle_deg) or not 0 <= max_angle_deg < 90:
        raise ValueError("max_angle_deg must be in [0, 90) degrees")
    t = np.arange(n_samples) * dt_s
    catalog = enumerate_events(
        model, variant, float(t[-1]) + margin, max_reflections=max_reflections, max_paths=max_paths
    )
    traces = np.zeros((offsets.size, n_samples))
    shape = (len(catalog.events), offsets.size)
    support = np.zeros(shape, dtype=bool)
    arrivals = np.full(shape, np.nan)
    params = np.full(shape, np.nan)
    valid = np.ones(traces.shape, dtype=bool)
    groups: dict[tuple[int, ...], list[int]] = {}
    for j, event in enumerate(catalog.events):
        groups.setdefault(event.traversal_counts, []).append(j)
    for indices in groups.values():
        events = [catalog.events[j] for j in indices]
        p, times, ok = event_moveout(model, events[0], offsets, max_angle_deg)
        for j in indices:
            support[j] = ok
            arrivals[j, ok] = times[ok]
            params[j, ok] = p[ok]
        # Aggregate ONLY after summing all distinct path coefficient products.
        amplitudes = path_amplitudes(model, events, p[ok]).sum(axis=0)
        traces[ok] += amplitudes[:, None] * np.asarray(
            ricker_wavelet(t[None, :] - times[ok, None], f_dom_hz)
        )
        for i in np.flatnonzero(~ok):
            if np.isfinite(times[i]):
                valid[i, np.abs(t - times[i]) <= margin] = False
            else:
                valid[i] = False
    p_max = min(
        float(slowness_from_angle(max_angle_deg, model.vp_m_s[0])),
        float(1 / np.max(model.vp_m_s)) * (1 - 1e-6),
    )
    return RayGather(traces, catalog, support, arrivals, params, valid, p_max, margin)
