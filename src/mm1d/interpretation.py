"""Finite-spread hyperbola fits, distinct from small-spread path RMS anchors.

Fits minimize equally weighted arrival-time residuals, not semblance and not
squared-time residuals. Resolution envelopes are deterministic timing budgets,
NOT statistical confidence intervals on noiseless synthetic data.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from mm1d.events import Event
from mm1d.model import LayeredModel
from mm1d.ray import event_moveout


@dataclass(frozen=True)
class StackingFit:
    """Free-intercept time-domain fit on a fixed set of offsets.

    Times/residuals in s, velocity in m/s, offset extent in m. fold is the
    actual number of arrivals, not the number of requested offset samples.
    This is a geometric best fit, not an amplitude-weighted coherence pick.
    """

    t0_s: float
    velocity_m_s: float
    rms_residual_s: float
    max_residual_s: float
    fold: int
    max_offset_m: float


@dataclass(frozen=True)
class MoveoutResolution:
    """Exact max-time-error envelope on a supplied velocity grid (m/s).

    feasible marks velocities admitting an intercept with every arrival
    within tolerance_s (s). lower_t0_s/upper_t0_s bound those intercepts;
    infeasible entries are NaN. Edge hits are scan-censored, not finite
    uncertainty bounds. Fold/offset membership is held FIXED for this test.
    """

    velocity_m_s: NDArray[np.float64]
    lower_t0_s: NDArray[np.float64]
    upper_t0_s: NDArray[np.float64]
    feasible: NDArray[np.bool_]
    tolerance_s: float


def _arrivals(
    offsets_m: NDArray[np.float64], arrivals_s: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, t = np.asarray(offsets_m, dtype=np.float64), np.asarray(arrivals_s, dtype=np.float64)
    if (
        x.ndim != 1
        or t.shape != x.shape
        or x.size < 3
        or np.any(~np.isfinite(x))
        or np.any(~np.isfinite(t))
        or np.any(t <= 0)
        or np.unique(np.abs(x)).size < 3
    ):
        raise ValueError("need >=3 distinct absolute offsets and finite positive arrivals (m, s)")
    return x, t


def fit_stacking_velocity(
    offsets_m: NDArray[np.float64], arrivals_s: NDArray[np.float64]
) -> StackingFit:
    """Fit sqrt(T0**2 + x**2/V**2) to fixed supported arrivals.

    Parameters
    ----------
    offsets_m : ndarray
        Source-receiver offsets, m. At least three distinct absolute offsets.
    arrivals_s : ndarray
        Finite positive exact or measured arrivals, s, same 1-D shape.
        Caller must exclude unsupported/out-of-record arrivals first.

    Returns
    -------
    StackingFit
        Free T0 (s), V (m/s), residuals and fold. Both fit parameters are
        positive. Degenerate/unphysical fits raise rather than invent a pick.
    """
    x, t = _arrivals(offsets_m, arrivals_s)
    scale = float(np.max(np.abs(x)))
    z = (x / scale) ** 2
    slope, intercept = np.polyfit(z, t * t, 1)
    if slope <= 10 * np.finfo(float).eps * np.max(t * t) or intercept <= 0:
        raise ValueError("arrivals do not admit a positive hyperbola initialization")
    # Scaled second parameter k = x_max / V has units s, like T0.
    result = least_squares(
        lambda a: np.sqrt(a[0] ** 2 + a[1] ** 2 * z) - t,
        np.sqrt([intercept, slope]),
        bounds=(np.finfo(float).eps, np.inf),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    if not result.success:
        raise RuntimeError(f"hyperbola fit failed: {result.message}")
    residual = np.asarray(result.fun)
    return StackingFit(
        float(result.x[0]),
        float(scale / result.x[1]),
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(np.abs(residual))),
        x.size,
        scale,
    )


def moveout_resolution(
    offsets_m: NDArray[np.float64],
    arrivals_s: NDArray[np.float64],
    velocity_grid_m_s: NDArray[np.float64],
    tolerance_s: float,
) -> MoveoutResolution:
    """Profile free intercepts within an explicit maximum timing error budget.

    offsets_m (m) and arrivals_s (s) are fixed supported arrivals as in
    fit_stacking_velocity. velocity_grid_m_s is finite, positive, strictly
    increasing (m/s). tolerance_s is a positive timing budget (s), supplied
    by the caller, e.g. sampling + a declared fraction of a wavelet period.
    No probability, noise model, or unique raw semblance argmax is assumed.

    For each V intersect all intervals
    (max(0,T_i-epsilon))**2 - x_i**2/V**2 <= T0**2
        <= (T_i+epsilon)**2 - x_i**2/V**2.
    This nonlinear envelope includes T0/V trade-offs without linearization.
    """
    x, t = _arrivals(offsets_m, arrivals_s)
    v = np.asarray(velocity_grid_m_s, dtype=np.float64)
    if (
        v.ndim != 1
        or v.size < 2
        or np.any(~np.isfinite(v))
        or np.any(v <= 0)
        or np.any(np.diff(v) <= 0)
    ):
        raise ValueError("velocity grid must be positive, finite, strictly increasing (m/s)")
    if not np.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError("tolerance_s must be finite and positive (s)")
    moveout = (x[None, :] / v[:, None]) ** 2
    lower = np.maximum(0.0, np.max(np.maximum(0, t - tolerance_s) ** 2 - moveout, axis=1))
    upper = np.min((t + tolerance_s) ** 2 - moveout, axis=1)
    feasible = (upper >= lower) & (upper > 0)
    return MoveoutResolution(
        v.copy(),
        np.sqrt(np.where(feasible, lower, np.nan)),
        np.sqrt(np.where(feasible, upper, np.nan)),
        feasible,
        tolerance_s,
    )


def finite_spread_trajectory(
    model: LayeredModel,
    event: Event,
    offsets_m: NDArray[np.float64],
    spread_limits_m: NDArray[np.float64],
    *,
    max_angle_deg: float = 40.0,
    usable: NDArray[np.bool_] | None = None,
) -> tuple[StackingFit | None, ...]:
    """Fit an event on successive cumulative spreads, separately from its RMS.

    offsets_m and increasing spread_limits_m are in m. max_angle_deg is the
    surface aperture in degrees. Optional usable is a bool mask over offsets
    (e.g. full in-record gates with valid samples). Each fit uses only rays
    supported by the aperture AND usable AND abs(x)<=spread limit. None
    denotes fewer than three distinct absolute offsets. Repeated end points
    expose support-limited plateaus; unsupported rays are never extrapolated.
    """
    x = np.asarray(offsets_m, dtype=np.float64)
    # event_moveout validates offsets, aperture and event/model consistency.
    _, times, supported = event_moveout(model, event, x, max_angle_deg)
    limits = np.asarray(spread_limits_m, dtype=np.float64)
    if (
        limits.ndim != 1
        or not limits.size
        or np.any(~np.isfinite(limits))
        or np.any(limits <= 0)
        or np.any(np.diff(limits) <= 0)
    ):
        raise ValueError("spread limits must be finite, positive and increasing (m)")
    valid = np.ones(x.shape, dtype=bool) if usable is None else np.asarray(usable)
    if valid.shape != x.shape or valid.dtype != np.bool_:
        raise ValueError("usable must be a bool mask over offsets")
    fits: list[StackingFit | None] = []
    for limit in limits:
        ok = supported & valid & np.isfinite(times) & (np.abs(x) <= limit)
        fits.append(
            fit_stacking_velocity(x[ok], times[ok]) if np.unique(np.abs(x[ok])).size >= 3 else None
        )
    return tuple(fits)
