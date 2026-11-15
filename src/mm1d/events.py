"""Bounded acoustic ray paths, independent of the omega-p recursion.

Interface k is the base of finite layer k; -1 is the pressure-release surface.
Catalogs retain distinct orderings even when their moveout is identical.
Synthesis never prunes by pressure amplitude (transmissions can exceed one).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from mm1d.model import LayeredModel, acoustic_reflection_p, acoustic_transmission_p
from mm1d.reflectivity import FREE_SURFACE_REFLECTIVITY, WavefieldVariant, variant_multiples


@dataclass(frozen=True)
class Event:
    """Immutable path record; times in s, velocity in m/s, counts dimensionless.

    surface_order counts surface reflections, internal_order counts underside
    reflections; total_reflections includes every turn except source/receiver.
    """

    turning_points: tuple[int, ...]
    traversal_counts: tuple[int, ...]
    t0_s: float
    vrms_m_s: float
    family: str
    surface_order: int
    internal_order: int

    @property
    def total_reflections(self) -> int:
        """Number of interface and surface reflections, receiver excluded."""
        return len(self.turning_points) - 2


@dataclass(frozen=True)
class EventCatalog:
    """Finite synthesis catalog and bounds reached; not an infinite response.

    max_time_s bounds zero-offset centers (s), not oblique arrival times.
    A reflection-bound hit means potentially in-window paths were omitted.
    Resource overflow raises instead of returning a silently partial catalog.
    """

    events: tuple[Event, ...]
    reached_bounds: tuple[str, ...]
    max_time_s: float
    max_reflections: int
    states_visited: int


class PathLimitError(RuntimeError):
    """Enumeration exceeded its path/state resource budget."""


def make_event(model: LayeredModel, turning_points: Sequence[int]) -> Event:
    """Validate a surface-to-surface path and compute T0 and RMS velocity.

    Parameters
    ----------
    model : LayeredModel
        Finite thicknesses (m), velocities (m/s), densities (kg/m³).
    turning_points : sequence of int
        Alternating down/up turning points, starting/ending at -1 (surface).

    Returns
    -------
    Event
        Validated path; transmission interactions are implicit in its legs.
    """
    points = tuple(turning_points)
    if len(points) < 3 or points[0] != -1 or points[-1] != -1:
        raise ValueError("event must start/end at surface (-1), with at least one reflector")
    if any(not isinstance(k, (int, np.integer)) or isinstance(k, bool) for k in points):
        raise ValueError("turning points must be integers")
    if any(k < -1 or k >= model.n_interfaces for k in points):
        raise ValueError("turning point out of range")
    counts = np.zeros(model.n_interfaces, dtype=np.int64)
    internal = 0
    for j, (a, b) in enumerate(pairwise(points)):
        if (b - a) * (1 if j % 2 == 0 else -1) <= 0:
            raise ValueError("reflection directions must alternate down/up")
        counts[min(a, b) + 1 : max(a, b) + 1] += 1
        if j % 2 == 1 and b != -1:
            internal += 1
    surface = points[1:-1].count(-1)
    family = (
        "primary"
        if len(points) == 3
        else ("mixed" if surface and internal else "surface" if surface else "interbed")
    )
    leg_times = counts * model.h_m / model.vp_m_s[:-1]
    t0 = float(leg_times.sum())
    vrms = float(np.sqrt(np.sum(leg_times * model.vp_m_s[:-1] ** 2) / t0))
    return Event(
        tuple(int(k) for k in points),
        tuple(int(n) for n in counts),
        t0,
        vrms,
        family,
        surface,
        internal,
    )


def path_amplitude(model: LayeredModel, event: Event, p_s_m: float = 0.0) -> float:
    """Pressure R/T path product at shared subcritical slowness p (s/m).

    Downward reflections use R, underside reflections -R; each crossed
    interface uses directional pressure T. Source and receiver have no R_s.
    No spreading, obliquity multiplier, or energy-normalized transmission.
    """
    if not np.isfinite(p_s_m) or p_s_m < 0 or p_s_m >= 1.0 / np.max(model.vp_m_s):
        raise ValueError("p_s_m must be finite, nonnegative and globally subcritical (s/m)")
    # Validate against this model rather than trusting foreign/manual records.
    checked = make_event(model, event.turning_points)
    amplitude = 1.0
    for j, (a, b) in enumerate(pairwise(checked.turning_points)):
        down = b > a
        for i in range(min(a, b) + 1, max(a, b)):
            td, tu = acoustic_transmission_p(
                model.vp_m_s[i],
                model.rho_kg_m3[i],
                model.vp_m_s[i + 1],
                model.rho_kg_m3[i + 1],
                p_s_m,
            )
            amplitude *= float(np.real(td if down else tu))
        if j == len(checked.turning_points) - 2:
            break  # receiver return, before any surface reflection
        if b == -1:
            amplitude *= FREE_SURFACE_REFLECTIVITY
        else:
            r = acoustic_reflection_p(
                model.vp_m_s[b],
                model.rho_kg_m3[b],
                model.vp_m_s[b + 1],
                model.rho_kg_m3[b + 1],
                p_s_m,
            )
            amplitude *= float(np.real(r)) * (1 if down else -1)
    return amplitude


def path_amplitudes(
    model: LayeredModel,
    events: Sequence[Event],
    p_s_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vector pressure products, shape (event, p); slownesses in s/m.

    Evaluates interface optics once per p, then applies each path's directional
    interactions independently. Retains every ordering and its multiplicity.
    """
    p = np.asarray(p_s_m, dtype=np.float64)
    if (
        p.ndim != 1
        or np.any(~np.isfinite(p))
        or np.any(p < 0)
        or np.any(p >= 1 / np.max(model.vp_m_s))
    ):
        raise ValueError("p_s_m must be a 1-D globally subcritical slowness array (s/m)")
    r = np.empty((model.n_interfaces, p.size))
    td, tu = np.empty_like(r), np.empty_like(r)
    for k in range(model.n_interfaces):
        args = (model.vp_m_s[k], model.rho_kg_m3[k], model.vp_m_s[k + 1], model.rho_kg_m3[k + 1], p)
        r[k] = np.real(acoustic_reflection_p(*args))
        down, up = acoustic_transmission_p(*args)
        td[k], tu[k] = np.real(down), np.real(up)
    amplitudes = np.ones((len(events), p.size))
    for row, event in enumerate(events):
        checked = make_event(model, event.turning_points)
        for j, (a, b) in enumerate(pairwise(checked.turning_points)):
            down = b > a
            for k in range(min(a, b) + 1, max(a, b)):
                amplitudes[row] *= td[k] if down else tu[k]
            if j == len(checked.turning_points) - 2:
                break
            amplitudes[row] *= FREE_SURFACE_REFLECTIVITY if b == -1 else r[b] * (1 if down else -1)
    return amplitudes


def enumerate_events(
    model: LayeredModel,
    variant: WavefieldVariant,
    max_time_s: float,
    *,
    max_reflections: int = 32,
    max_paths: int = 100_000,
) -> EventCatalog:
    """Enumerate alternating paths bounded by zero-offset time and reflections.

    Parameters
    ----------
    model : LayeredModel
        Acoustic model (SI units).
    variant : WavefieldVariant
        full, no_surface, or primaries; switches shared with reflectivity.
    max_time_s : float
        Latest zero-offset event center, s. Include a wavelet-tail margin when
        using the catalog for synthesis. T(p) >= T0 makes this a safe ray bound.
    max_reflections : int, optional
        Total surface + interface turns, excluding source/receiver (default 32).
    max_paths : int, optional
        Maximum returned paths AND visited partial states (default 100000).

    Returns
    -------
    EventCatalog
        Distinct coherent paths, sorted by T0 then turning points. Bounds are
        reported; no claim of order convergence is implicit in a capped set.
    """
    surface_on, internal_on = variant_multiples(variant)
    if not np.isfinite(max_time_s) or max_time_s <= 0:
        raise ValueError("max_time_s must be finite and > 0 (s)")
    for name, value in (("max_reflections", max_reflections), ("max_paths", max_paths)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    depths_s = np.concatenate(([0.0], np.cumsum(model.h_m / model.vp_m_s[:-1])))
    # Each state ends at an already reflected turning point, ready to travel.
    states: list[tuple[tuple[int, ...], float]] = [((-1,), 0.0)]
    events: list[Event] = []
    reached: set[str] = set()
    visited = 0
    while states:
        points, elapsed = states.pop()
        visited += 1
        if visited > max_paths:
            raise PathLimitError(
                f"max_paths={max_paths} exceeded by partial states; tighten time/order bounds"
            )
        a = points[-1]
        down = len(points) % 2 == 1
        targets = range(a + 1, model.n_interfaces) if down else range(-1, a)
        for b in targets:
            time = elapsed + abs(float(depths_s[b + 1] - depths_s[a + 1]))
            # Earliest possible return from b: vertical travel straight up.
            if time + depths_s[b + 1] > max_time_s + 1e-12:
                reached.add("time")
                continue
            new = (*points, b)
            if b == -1:
                events.append(make_event(model, new))
                if len(events) > max_paths:
                    raise PathLimitError(f"max_paths={max_paths} exceeded by completed paths")
                if not surface_on:
                    continue
            elif not down and not internal_on:
                continue
            # b must reflect to continue (including an intermediate surface).
            if len(new) - 1 > max_reflections:
                reached.add("reflections")
                continue
            if len(states) + visited >= max_paths:
                raise PathLimitError(f"max_paths={max_paths} exceeded by pending partial states")
            states.append((new, time))
    events.sort(key=lambda event: (event.t0_s, event.turning_points))
    return EventCatalog(tuple(events), tuple(sorted(reached)), max_time_s, max_reflections, visited)


def display_events(
    model: LayeredModel,
    catalog: EventCatalog,
    *,
    max_order: int = 3,
    min_amplitude: float = 0.01,
) -> tuple[Event, ...]:
    """Human-scale selection only; never use this amplitude heuristic for synthesis.

    max_order counts surface + underside bounces; min_amplitude is an absolute
    dimensionless normal-incidence pressure threshold, not an oblique bound.
    """
    if not isinstance(max_order, int) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")
    if not np.isfinite(min_amplitude) or min_amplitude < 0:
        raise ValueError("min_amplitude must be finite and nonnegative")
    return tuple(
        event
        for event in catalog.events
        if event.surface_order + event.internal_order <= max_order
        and abs(path_amplitude(model, event)) >= min_amplitude
    )
