"""Auditable energetic-cell attribution for bounded diagnostic ray gathers.

Coherence alone is scale free: a strong ridge requires a declared gate-energy
threshold as well as coherence and fold. Attribution uses coherent sums of
coincident orderings, never |A(0)| display pruning. It is a synthesis accounting
check, not an independent validation of the path physics (see moveout tests).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mm1d.events import Event, path_amplitudes
from mm1d.model import LayeredModel
from mm1d.ray import RayGather
from mm1d.semblance import SemblanceResult, _hyperbola_gate_samples
from mm1d.wavelet import ricker_wavelet


@dataclass(frozen=True)
class RidgeAttribution:
    """Accounting for selected (velocity_index, time_index) cells.

    group_energy has shape (coincident group, selected cell), in squared
    pressure summed over the ACTUAL scan gates/traces. observed_energy has
    shape (selected cell,). Their difference is the signed interference
    cross energy, not unexplained energy. groups retain every path ordering.
    dominant_center_shift_s is the mean trial-center minus dominant-event
    arrival over actual contributing AND event-supported traces (s).
    dominant_differential_error_s is the max residual after removing that
    common shift (s); dominant_event_fold counts those traces. These expose
    tail/gate shifts separately from differential moveout misalignment.
    reconstruction_error is the max samplewise pressure mismatch against the
    supplied ray gather; this tests accounting, not independent correctness.
    """

    groups: tuple[tuple[Event, ...], ...]
    velocity_index: NDArray[np.int64]
    time_index: NDArray[np.int64]
    group_energy: NDArray[np.float64]
    observed_energy: NDArray[np.float64]
    reconstruction_error: float
    dominant_center_shift_s: NDArray[np.float64]
    dominant_differential_error_s: NDArray[np.float64]
    dominant_event_fold: NDArray[np.int64]


def strong_ridge_cells(
    scan: SemblanceResult,
    *,
    coherence_min: float = 0.8,
    relative_rms_min: float = 0.01,
    min_fold: int = 3,
) -> NDArray[np.bool_]:
    """Select energetic, coherent, supported cells using explicit thresholds.

    coherence_min and relative_rms_min are dimensionless, in (0,1]. Gate RMS
    is relative to this panel's maximum gate RMS, not the strongest catalog
    path. min_fold is an integer >=2. These are declared interpretation
    thresholds, not synthesis pruning or universal detectability claims.
    """
    if not 0 < coherence_min <= 1 or not 0 < relative_rms_min <= 1:
        raise ValueError("coherence/RMS thresholds must be in (0, 1]")
    if not isinstance(min_fold, int) or isinstance(min_fold, bool) or min_fold < 2:
        raise ValueError("min_fold must be an integer >=2")
    return (
        (scan.panel >= coherence_min)
        & (scan.fold >= min_fold)
        & (scan.gate_rms >= relative_rms_min * np.max(scan.gate_rms))
        & (scan.gate_rms > 0)
    )


def attribute_ridge_cells(
    model: LayeredModel,
    ray: RayGather,
    offsets_m: NDArray[np.float64],
    dt_s: float,
    f_dom_hz: float,
    t0_grid_s: NDArray[np.float64],
    velocity_grid_m_s: NDArray[np.float64],
    selected: NDArray[np.bool_],
    gate_s: float,
) -> RidgeAttribution:
    """Decompose selected noiseless ray scan cells into coincident path groups.

    Parameters
    ----------
    model, ray : LayeredModel, RayGather
        Model and unmodified diagnostic ray gather with a bounded catalog.
    offsets_m : ndarray
        Offset grid, m, matching ray traces.
    dt_s, f_dom_hz : float
        Gather sampling (s) and Ricker frequency (Hz), as used for synthesis.
    t0_grid_s, velocity_grid_m_s : ndarray
        Scan coordinates, s and m/s.
    selected : ndarray of bool
        Shape (velocity, T0), e.g. strong_ridge_cells output. All selected
        cells are accounted for, including tails, ties and mixtures.
    gate_s : float
        Same centered gate duration as the scan, s.

    Returns
    -------
    RidgeAttribution
        Coherent group energies with exact scan support/interpolation. No
        normalized coherence ridge is automatically declared a distinct event.
        Reflection-truncated catalogs are rejected for this report.
    """
    if "reflections" in ray.catalog.reached_bounds:
        raise ValueError("establish catalog reflection-order convergence before attribution")
    x = np.asarray(offsets_m, dtype=np.float64)
    t0 = np.asarray(t0_grid_s, dtype=np.float64)
    v = np.asarray(velocity_grid_m_s, dtype=np.float64)
    mask = np.asarray(selected)
    if mask.dtype != np.bool_ or mask.shape != (v.size, t0.size):
        raise ValueError("selected must be a bool mask on the (velocity, T0) grid")
    if x.shape != (ray.traces.shape[0],) or np.any(~np.isfinite(x)):
        raise ValueError("offsets must match the ray gather")
    if (
        not np.isfinite(dt_s)
        or dt_s <= 0
        or not np.isfinite(gate_s)
        or gate_s <= 0
        or not np.isfinite(f_dom_hz)
        or f_dom_hz <= 0
        or t0.ndim != 1
        or v.ndim != 1
        or np.any(~np.isfinite(t0))
        or np.any(~np.isfinite(v))
        or np.any(v <= 0)
    ):
        raise ValueError("invalid sampling, scan grids, frequency or gate")
    iv, it = np.asarray(np.nonzero(mask), dtype=np.int64)
    gate_count = max(1, round(gate_s / dt_s))
    gate_times = (np.arange(gate_count, dtype=np.float64) - (gate_count - 1) / 2) * dt_s
    p_cut = ray.p_max_s_m * (1 - 1e-9)
    grouped: dict[tuple[int, ...], list[int]] = {}
    for j, event in enumerate(ray.catalog.events):
        grouped.setdefault(event.traversal_counts, []).append(j)
    indices = list(grouped.values())
    groups = tuple(tuple(ray.catalog.events[j] for j in rows) for rows in indices)
    if not groups and iv.size:
        raise ValueError("cannot attribute selected cells with an empty event catalog")
    energy = np.zeros((len(groups), iv.size))
    observed = np.zeros(iv.size)
    reconstructed = np.zeros_like(ray.traces)
    time = np.arange(ray.traces.shape[1]) * dt_s

    def cell_energy(traces: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.empty(iv.size)
        for start in range(0, iv.size, 128):
            block = slice(start, start + 128)
            values, _ = _hyperbola_gate_samples(
                traces,
                x,
                dt_s,
                t0[it[block]],
                v[iv[block]],
                gate_times,
                p_cut,
                ray.valid_samples,
            )
            result[block] = np.sum(values**2, axis=(1, 2))
        return result

    observed[:] = cell_energy(ray.traces)
    for k, rows in enumerate(indices):
        j = rows[0]
        ok = ray.supported[j]
        amplitudes = path_amplitudes(model, groups[k], ray.slownesses_s_m[j, ok]).sum(axis=0)
        component = np.zeros_like(ray.traces)
        component[ok] = amplitudes[:, None] * np.asarray(
            ricker_wavelet(time[None, :] - ray.arrival_times_s[j, ok, None], f_dom_hz)
        )
        reconstructed += component
        energy[k] = cell_energy(component)
    shift, differential = np.zeros(iv.size), np.zeros(iv.size)
    event_fold = np.zeros(iv.size, dtype=np.int64)
    dominant = np.argmax(energy, axis=0) if len(groups) else np.zeros(iv.size, dtype=int)
    for start in range(0, iv.size, 128):
        block = slice(start, start + 128)
        _, trace_ok = _hyperbola_gate_samples(
            ray.traces,
            x,
            dt_s,
            t0[it[block]],
            v[iv[block]],
            gate_times,
            p_cut,
            ray.valid_samples,
        )
        trial = np.sqrt(t0[it[block], None] ** 2 + (x[None, :] / v[iv[block], None]) ** 2)
        for local, cell in enumerate(range(start, min(start + 128, iv.size))):
            j = indices[int(dominant[cell])][0]
            ok = trace_ok[local] & ray.supported[j]
            event_fold[cell] = int(ok.sum())
            if ok.any():
                difference = trial[local, ok] - ray.arrival_times_s[j, ok]
                shift[cell] = float(difference.mean())
                differential[cell] = float(np.max(np.abs(difference - shift[cell])))
    return RidgeAttribution(
        groups,
        iv,
        it,
        energy,
        observed,
        float(np.max(np.abs(reconstructed - ray.traces))),
        shift,
        differential,
        event_fold,
    )
