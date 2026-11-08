"""Offset-gather physics validation against analytic anchors (M4).

Anchor logic (PLAN.md M4 acceptance, corrected 2026-11-09, + AGENTS.md
conventions):

- **Snell ray tracing** (:func:`mm1d.gathers.path_offset_time`,
  :func:`mm1d.gathers.primary_offset_times`): exact closed forms per bounce
  path -- ``x = sum h tan(theta)``, ``T = sum h / (v cos(theta))`` over the
  crossed legs with ``p = sin(theta)/v`` constant.  T1 primary ``2h tan(theta)``
  closed form; T2 Snell refraction into the sediment; the T2 water-bottom
  multiple as 4 water legs; NaN where a leg is evanescent.
- **Offset-gather synthesis** (:func:`mm1d.gathers.offset_gather`): the
  rho-filtered inverse slant stack (mirrored ``cos`` kernel,
  ``|w|/pi`` normalization).  Event *times* land on the ray-traced moveout
  ``T(x(p*))`` of their stationary ray; amplitudes follow the 2-D
  line-source stationary-phase scale ``A(p*) / sqrt(2 pi |dx/dp|)`` in the
  ``sqrt(w)``-reshaped wavelet measure (continuous through ``x = 0``);
  stationary rays near the aperture edge lose amplitude (Fresnel truncation,
  documented, not anchored).
- **Multiple-velocity expectation (corrected)**: the free-surface water-bottom
  multiple is pinned at the water velocity (``V_stk = v_w``, steeper moveout,
  the radon-separability signature) -- *lower* apparent velocity than a
  primary at the same ``T0``; its picks must follow its own hyperbola, not
  the same-``T0`` primary hyperbola.
- **Round trip** (:func:`mm1d.gathers.slant_stack`): the one-sided forward
  tau-p transform of a synthesized gather recovers the original angle gather
  eventwise (times exact; correlation/scale within finite-aperture
  corrections of order 10 %), away from mutes and aperture edges.
"""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import hilbert

from mm1d.gathers import (
    SURFACE,
    _aperture_weights,
    _refine_angles,
    angle_gather,
    critical_mute_slowness,
    default_offset_grid,
    layer_angles,
    offset_gather,
    offset_impulse_gather,
    path_offset_time,
    primary_offset_times,
    slant_stack,
    slowness_from_angle,
)
from mm1d.reflectivity import _stack_terms
from mm1d.wavelet import ricker_spectrum, ricker_wavelet

DT_S = 2.0e-3
"""Default sample interval (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Dominant Ricker frequency (PLAN.md numerical defaults)."""

ANGLES_1DEG = np.arange(0.0, 40.0 + 1.0e-9, 1.0)
"""Default gather-angle grid from PLAN.md (0-40 deg, 1 deg step)."""

FFT_OVERSAMPLE = 8
"""FFT record extension factor (M2/M3 practice: late multiples wrap outside)."""

WB_MULTIPLE_PATH = [SURFACE, 0, SURFACE, 0, SURFACE]
"""First-order water-bottom multiple: down/up, surface bounce, down/up."""


def envelope(trace: NDArray[np.float64] | np.float64) -> NDArray[np.float64]:
    """Analytic-signal envelope of a trace (accepts scalar-or-array row types)."""
    return np.abs(hilbert(np.atleast_1d(np.asarray(trace, dtype=np.float64))))


def pick_in_window(
    env: NDArray[np.float64] | np.float64, t0_s: float, half_width_s: float = 0.06
) -> float:
    """Time of the envelope maximum within +-``half_width_s`` of ``t0_s``."""
    env = np.atleast_1d(np.asarray(env, dtype=np.float64))
    i0 = round(float(t0_s) / DT_S)
    width = round(half_width_s / DT_S)
    lo = max(0, i0 - width)
    hi = min(env.size, i0 + width + 1)
    if hi <= lo:
        raise ValueError(f"pick window around {t0_s} s is outside the trace")
    return float((lo + int(np.argmax(env[lo:hi]))) * DT_S)


# -- layer_angles: Snell refraction ----------------------------------------------


def test_layer_angles_snell_refraction(t2):
    angles = np.array([0.0, 10.0, 20.0, 29.0])
    got = layer_angles(t2, angles)
    sin_water = np.sin(np.radians(angles))
    expected_sediment = np.degrees(np.arcsin(sin_water * 2400.0 / 1500.0))
    np.testing.assert_allclose(got[0], angles, atol=1e-12)
    np.testing.assert_allclose(got[1], expected_sediment, atol=1e-10)
    # zero incidence stays zero everywhere
    assert got[0, 0] == 0.0 and got[1, 0] == 0.0


def test_layer_angles_evanescent_nan(t2):
    got = layer_angles(t2, np.array([39.0, 40.0]))
    # water layer always propagating; sediment critical at 38.68 deg, half-space at 30 deg
    assert np.all(np.isfinite(got[0]))
    assert np.all(np.isnan(got[1]))
    assert np.all(np.isnan(got[2]))


# -- path_offset_time: closed forms ----------------------------------------------


@pytest.mark.parametrize(
    ("theta_deg",),
    [(0.0,), (10.0,), (20.0,), (29.0,), (40.0,)],
)
def test_path_offset_time_t1_primary_closed_form(t1, theta_deg):
    x, t = path_offset_time(t1, [SURFACE, 0, SURFACE], np.array([theta_deg]))
    tan_theta = np.tan(np.radians(theta_deg))
    np.testing.assert_allclose(x, 2.0 * 500.0 * tan_theta, rtol=1e-12)
    np.testing.assert_allclose(
        t, 2.0 * 500.0 / (2000.0 * np.cos(np.radians(theta_deg))), rtol=1e-12
    )


def test_path_offset_time_zero_offset_matches_twt(t2, t3):
    angles = np.array([0.0])
    # P2 of T2: 0.4 + 0.5 = 0.9 s; water-bottom multiple: 4 x 0.2 = 0.8 s
    _, t_p2 = path_offset_time(t2, [SURFACE, 1, SURFACE], angles)
    _, t_mult = path_offset_time(t2, WB_MULTIPLE_PATH, angles)
    assert t_p2[0] == pytest.approx(0.9, abs=1e-12)
    assert t_mult[0] == pytest.approx(0.8, abs=1e-12)
    # T3 interbed reverberation between interfaces 1 and 2 (inside layer 2,
    # TWT 0.3 s): primary-to-int-2 round trip (1.0 s) + 2 internal legs (0.6 s)
    _, t_ib = path_offset_time(t3, [SURFACE, 2, 1, 2, SURFACE], angles)
    assert t_ib[0] == pytest.approx(1.6, abs=1e-12)
    # every path at zero incidence has zero offset
    for path in ([SURFACE, 1, SURFACE], WB_MULTIPLE_PATH, [SURFACE, 2, 1, 2, SURFACE]):
        x, _ = path_offset_time(t2 if path is WB_MULTIPLE_PATH else t3, path, angles)
        assert x[0] == pytest.approx(0.0, abs=1e-12)


def test_path_offset_time_t2_snell_manual(t2):
    angles = np.array([0.0, 7.0, 15.0, 25.0, 29.0])
    h_w, v_w, h_s, v_s = 300.0, 1500.0, 600.0, 2400.0
    sin_w = np.sin(np.radians(angles))
    theta_s = np.arcsin(sin_w * v_s / v_w)
    x_expected = 2.0 * (h_w * np.tan(np.radians(angles)) + h_s * np.tan(theta_s))
    t_expected = 2.0 * (h_w / (v_w * np.cos(np.radians(angles))) + h_s / (v_s * np.cos(theta_s)))
    x, t = path_offset_time(t2, [SURFACE, 1, SURFACE], angles)
    np.testing.assert_allclose(x, x_expected, rtol=1e-12)
    np.testing.assert_allclose(t, t_expected, rtol=1e-12)


def test_path_offset_time_water_bottom_multiple_four_legs(t2):
    angles = ANGLES_1DEG
    x, t = path_offset_time(t2, WB_MULTIPLE_PATH, angles)
    np.testing.assert_allclose(x, 4.0 * 300.0 * np.tan(np.radians(angles)), rtol=1e-12)
    np.testing.assert_allclose(t, 4.0 * 300.0 / (1500.0 * np.cos(np.radians(angles))), rtol=1e-12)
    # pure water path: still a real ray at 40 deg (sediment/half-space never entered)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(t))


def test_path_offset_time_t3_pegleg_manual(t3):
    """Peg-leg water bounce + P2 path: independent per-leg recomputation."""
    angles = np.array([3.0, 12.0, 21.0, 33.0])
    vp = t3.vp_m_s
    h = t3.h_m
    p = slowness_from_angle(angles, vp[0])
    sin = [p * v for v in vp]
    x_expected = np.zeros_like(p)
    t_expected = np.zeros_like(p)
    # path [-1, 1, -1, 0, -1]: one P2 round trip (layers 0,1) + one WB bounce (layer 0)
    for lo, hi in ((0, 1), (0, 1), (0, 0), (0, 0)):
        for j in range(lo, hi + 1):
            theta_j = np.arcsin(sin[j])
            x_expected = x_expected + h[j] * np.tan(theta_j)
            t_expected = t_expected + h[j] / (vp[j] * np.cos(theta_j))
    x, t = path_offset_time(t3, [SURFACE, 1, SURFACE, 0, SURFACE], angles)
    np.testing.assert_allclose(x, x_expected, rtol=1e-12)
    np.testing.assert_allclose(t, t_expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("model_name", "path", "finite_deg", "nan_deg"),
    [
        ("t2", [SURFACE, 1, SURFACE], 38.0, 39.0),  # sediment critical at 38.68 deg
        ("t3", [SURFACE, 3, SURFACE], 36.0, 38.0),  # half-space critical at 36.87 deg
    ],
)
def test_path_offset_time_evanescent_nan(canonical_models, model_name, path, finite_deg, nan_deg):
    model = canonical_models[model_name]
    x, t = path_offset_time(model, path, np.array([finite_deg, nan_deg]))
    assert np.all(np.isfinite(x[[0]])) and np.all(np.isfinite(t[[0]]))
    assert np.isnan(x[1]) and np.isnan(t[1])


def test_path_offset_time_validation(t2):
    angles = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match="at least 2 entries"):
        path_offset_time(t2, [SURFACE], angles)
    with pytest.raises(ValueError, match="must differ"):
        path_offset_time(t2, [SURFACE, 0, 0, SURFACE], angles)
    with pytest.raises(ValueError, match="out of range"):
        path_offset_time(t2, [SURFACE, 2, SURFACE], angles)  # T2 has interfaces 0, 1
    with pytest.raises(ValueError, match="must be integers"):
        path_offset_time(t2, cast("list[int]", [SURFACE, 0.5, SURFACE]), angles)


def test_primary_offset_times_matches_paths(t1, t2, t3):
    for model in (t1, t2, t3):
        x, t = primary_offset_times(model, ANGLES_1DEG)
        assert x.shape == (ANGLES_1DEG.size, model.n_interfaces)
        # zero-offset row: x = 0, t = interface TWTs
        np.testing.assert_allclose(x[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(t[0], model.interface_twt_s, rtol=1e-12)
        # per-interface equality with the generic path function (subcritical angles)
        for k in range(model.n_interfaces):
            x_path, t_path = path_offset_time(model, [SURFACE, k, SURFACE], ANGLES_1DEG)
            np.testing.assert_allclose(x[:, k], x_path, rtol=1e-12)
            np.testing.assert_allclose(t[:, k], t_path, rtol=1e-12)
        # offsets increase monotonically with angle wherever defined
        valid = np.isfinite(x)
        for k in range(model.n_interfaces):
            column = x[valid[:, k], k]
            assert np.all(np.diff(column) > 0.0)


# -- default_offset_grid ----------------------------------------------------------


def test_default_offset_grid_extent(t1, t2):
    grid_t1 = default_offset_grid(t1, ANGLES_1DEG, n_offsets=31)
    assert grid_t1[0] == 0.0
    assert grid_t1[-1] == pytest.approx(2.0 * 500.0 * np.tan(np.radians(40.0)), rel=1e-12)
    # T2: the post-critical mute at 30 deg caps the aperture; farthest usable ray at 29 deg
    grid_t2 = default_offset_grid(t2, ANGLES_1DEG)
    x_p2, _ = path_offset_time(t2, [SURFACE, 1, SURFACE], np.array([29.0]))
    assert grid_t2[-1] == pytest.approx(float(x_p2[0]), rel=1e-12)


def test_default_offset_grid_validation(t1):
    with pytest.raises(ValueError, match="n_offsets"):
        default_offset_grid(t1, ANGLES_1DEG, n_offsets=1)
    with pytest.raises(ValueError, match="degenerate"):
        default_offset_grid(t1, np.array([0.0, 0.0]))


# -- offset-gather synthesis: event anchors ---------------------------------------


def sqrtomega_ricker_envelope_peak(n_fft: int) -> float:
    """Envelope peak of the sqrt(w)-reshaped unit Ricker (synthesis wavelet measure)."""
    freq = np.fft.rfftfreq(n_fft, d=DT_S)
    omega = 2.0 * np.pi * freq
    spectrum = np.sqrt(omega) * np.asarray(ricker_spectrum(freq, F_DOM_HZ)) / DT_S
    reference = np.fft.irfft(spectrum, n=n_fft)
    return float(np.max(envelope(reference)))


def test_t1_offset_gather_primary_hyperbola_times(t1):
    """The T1 primary lands on the closed-form hyperbola at every stationary offset."""
    h, v = 500.0, 2000.0
    theta_test = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 38.0, 40.0])
    offsets = 2.0 * h * np.tan(np.radians(theta_test))
    times = 2.0 * h / (v * np.cos(np.radians(theta_test)))
    n_samples = round(1.5 / DT_S)
    gather = offset_gather(
        t1,
        "primaries",
        ANGLES_1DEG,
        offsets,
        DT_S,
        n_samples,
        F_DOM_HZ,
        n_fft=FFT_OVERSAMPLE * n_samples,
    )
    for x_theory, t_theory, trace in zip(offsets, times, gather, strict=True):
        pick = pick_in_window(envelope(trace), t_theory)
        assert abs(pick - t_theory) <= 1.5 * DT_S, f"x = {x_theory:.1f} m"


def test_t1_offset_gather_stationary_phase_amplitudes(t1):
    """Interior-stationary amplitudes match A(p)/sqrt(2 pi |dx/dp|) within 3 %.

    Anchored for stationary rays at least a Fresnel zone inside the tapered
    aperture (theta <= 15 deg on the default 0-40 deg grid).  Around 20 deg
    the anchor drifts a few % (interference with the aperture-taper skirt);
    asserted separately within 10 %.
    """
    h, v = 500.0, 2000.0
    theta_test = np.array([0.0, 5.0, 10.0, 15.0])
    offsets = 2.0 * h * np.tan(np.radians(theta_test))
    times = 2.0 * h / (v * np.cos(np.radians(theta_test)))
    n_samples = round(1.5 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    gather = offset_gather(
        t1, "primaries", ANGLES_1DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    p0 = sqrtomega_ricker_envelope_peak(n_fft)
    p = np.atleast_1d(np.asarray(slowness_from_angle(theta_test, v)))
    q = np.sqrt(1.0 / v**2 - p**2)
    dx_dp = 2.0 * h / (v**2 * q**3)
    reflection = np.array([float(np.real(t1.reflection(0, theta))) for theta in theta_test])
    predicted = reflection / np.sqrt(2.0 * np.pi * dx_dp) * p0
    for pred, t_theory, trace in zip(predicted, times, gather, strict=True):
        i0 = round(float(t_theory) / DT_S)
        got = float(np.max(envelope(np.asarray(trace))[i0 - 30 : i0 + 30]))
        assert got == pytest.approx(pred, rel=0.03)
    # 20 deg sits in the interference band with the taper skirt: looser bound
    theta_edge = np.array([20.0])
    offsets_edge = 2.0 * h * np.tan(np.radians(theta_edge))
    t_edge = 2.0 * h / (v * np.cos(np.radians(theta_edge)))
    p_edge = np.atleast_1d(np.asarray(slowness_from_angle(theta_edge, v)))
    dx_edge = 2.0 * h / (v**2 * np.sqrt(1.0 / v**2 - p_edge**2) ** 3)
    pred_edge = float(np.real(t1.reflection(0, 20.0))) / np.sqrt(2.0 * np.pi * dx_edge) * p0
    gather_edge = offset_gather(
        t1, "primaries", ANGLES_1DEG, offsets_edge, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    i_edge = round(float(t_edge[0]) / DT_S)
    got_edge = float(np.max(envelope(np.asarray(gather_edge[0]))[i_edge - 30 : i_edge + 30]))
    assert got_edge == pytest.approx(pred_edge, rel=0.10)


@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
def test_zero_offset_trace_preserved_at_x0_primaries(canonical_models, model_name):
    """At x = 0 the synthesized gather carries every primary at its exact TWT."""
    model = canonical_models[model_name]
    record_s = {"t1": 1.5, "t2": 3.0, "t3": 4.0}[model_name]
    n_samples = round(record_s / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    gather = offset_gather(
        model,
        "primaries",
        ANGLES_1DEG,
        np.array([0.0]),
        DT_S,
        n_samples,
        F_DOM_HZ,
        n_fft=n_fft,
    )
    env = envelope(gather[0])
    for t0 in model.interface_twt_s:
        pick = pick_in_window(env, float(t0))
        assert abs(pick - t0) <= 1.5 * DT_S, f"event at {t0:.3f} s"


def test_t2_full_zero_offset_train_alternates(t2):
    """x = 0 trace of the full variant: primaries + alternating water-bottom train."""
    n_samples = round(3.0 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    gather = offset_gather(
        t2, "full", ANGLES_1DEG, np.array([0.0]), DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    env = envelope(gather[0])
    for t0 in (0.4, 0.8, 0.9, 1.2):
        pick = pick_in_window(env, t0)
        assert abs(pick - t0) <= 1.5 * DT_S, f"event at {t0:.1f} s"
    # train present in full only: no_surface lacks it
    gather_ns = offset_gather(
        t2, "no_surface", ANGLES_1DEG, np.array([0.0]), DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    env_ns = envelope(gather_ns[0])
    window = slice(round(0.75 / DT_S), round(0.85 / DT_S))
    assert np.max(env_ns[window]) < 0.1 * np.max(env[window])


def test_t2_water_bottom_multiple_moveout_flatter_correction(t2):
    """WB multiple follows ITS hyperbola (V_stk = v_w), steeper than a same-T0 primary.

    Expectation corrected 2026-11-09: free-surface multiples sit at *lower*
    apparent velocity than same-T0 primaries (here 1500 vs ~1790 m/s), the
    radon-separability signature -- not above them.
    """
    n_samples = round(3.0 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    v_w = float(t2.vp_m_s[0])
    theta_probe = np.array([5.0, 15.0, 25.0])
    x_mult, t_mult = path_offset_time(t2, WB_MULTIPLE_PATH, theta_probe)
    gather = offset_gather(t2, "full", ANGLES_1DEG, x_mult, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft)
    v_rms_same_t0 = np.sqrt((0.4 * 1500.0**2 + 0.4 * 2400.0**2) / 0.8)
    assert v_rms_same_t0 == pytest.approx(2001.0, rel=5e-3)  # Dix primary at T0 = 0.8 s
    assert v_w < v_rms_same_t0  # the corrected expectation, stated for the record
    for x_theory, t_theory, trace in zip(x_mult, t_mult, gather, strict=True):
        pick = pick_in_window(envelope(trace), float(t_theory))
        assert abs(pick - t_theory) <= 1.5 * DT_S, f"x = {x_theory:.1f} m"
        # and the same-T0 primary hyperbola separates with offset (steeper multiple)
        t_primary_hyp = np.sqrt(0.8**2 + (x_theory / v_rms_same_t0) ** 2)
        if x_theory > 300.0:
            assert abs(t_primary_hyp - t_theory) > 0.010


def test_aperture_weights_mute_exclusion(t2):
    """Muted slowness gets exactly zero weight; the cut ends the integral."""
    p, weights = _aperture_weights(t2, ANGLES_1DEG, True, taper_fraction=0.0)
    p_mute = critical_mute_slowness(t2)
    assert p_mute is not None  # T2 has a fastest deeper layer (3000 m/s half-space)
    assert np.all(weights[p >= p_mute * (1.0 - 1.0e-9)] == 0.0)
    assert np.all(weights[p < p_mute * (1.0 - 1.0e-9)] > 0.0)
    # the last unmuted node keeps only its half-interval ending at the cut
    i_last = int(np.flatnonzero(weights > 0.0)[-1])
    assert weights[i_last] == pytest.approx((p_mute - p[i_last - 1]) / 2.0, rel=1e-6)
    with pytest.raises(ValueError, match="taper_fraction"):
        _aperture_weights(t2, ANGLES_1DEG, True, taper_fraction=1.0)


def test_offset_gather_causality_and_taper(t2):
    """Acausal radiation is suppressed to ~1 % by the aperture taper.

    Exact physics is causal: nothing may arrive before ``t = x / v_max``
    (fastest possible path).  The hard aperture truncation -- worst just
    below the post-critical mute, where ``|U|`` grows toward resonance --
    radiates acausally at ~3 % of peak without the taper; the default Tukey
    taper suppresses it (the M4 artifact-control acceptance).  Literal FFT
    wrap-around is separately negligible at the 8x record extension
    (measured < 3e-5 of peak).
    """
    n_samples = round(3.0 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    offsets = default_offset_grid(t2, ANGLES_1DEG)
    times = np.arange(n_samples) * DT_S
    v_max = float(np.max(t2.vp_m_s))

    def acausal_fraction(gather: np.ndarray) -> float:
        peak = float(np.max(np.abs(gather)))
        worst = 0.0
        for trace, x in zip(gather, offsets, strict=True):
            before = times < x / v_max - 0.04
            if before.any():
                worst = max(worst, float(np.max(np.abs(np.asarray(trace)[before]))) / peak)
        return worst

    tapered = offset_gather(
        t2, "full", ANGLES_1DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    untapered = offset_gather(
        t2,
        "full",
        ANGLES_1DEG,
        offsets,
        DT_S,
        n_samples,
        F_DOM_HZ,
        n_fft=n_fft,
        taper_fraction=0.0,
    )
    assert acausal_fraction(tapered) <= 0.015
    assert acausal_fraction(untapered) > 0.02  # the taper does the work
    # far-offset pre-event zone (beyond the near-field ramp): alias noise
    # controlled by the internal aperture refinement
    far = offsets > 600.0
    pre = np.abs(tapered[far][:, times < 0.30])
    assert float(np.max(pre)) <= 0.015 * float(np.max(np.abs(tapered)))


def test_offset_gather_refinement_convergence(t2):
    """The internal anti-aliasing refinement converges to an explicit fine grid."""
    n_samples = round(3.0 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    offsets = default_offset_grid(t2, ANGLES_1DEG)
    default_grid = offset_gather(
        t2, "primaries", ANGLES_1DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    fine_grid = offset_gather(
        t2,
        "primaries",
        np.arange(0.0, 40.0 + 1e-9, 0.125),
        offsets,
        DT_S,
        n_samples,
        F_DOM_HZ,
        n_fft=n_fft,
    )
    live = np.arange(n_samples) * DT_S <= 0.7 * n_samples * DT_S
    diff = np.abs(default_grid[:, live] - fine_grid[:, live])
    assert float(np.max(diff)) <= 0.02 * float(np.max(np.abs(fine_grid)))


def test_refine_angles_helper():
    """Angle densification keeps endpoints, monotonicity and the requested factor."""
    angles = np.array([0.0, 1.0, 3.0])
    refined = _refine_angles(angles, 4)
    assert refined[0] == 0.0 and refined[-1] == 3.0
    assert refined.size == 9
    assert np.all(np.diff(refined) > 0.0)
    assert _refine_angles(angles, 1) is angles


def test_stack_terms_even_in_slowness(t2):
    """Engine terms depend on p^2 only -- the cos-kernel justification.

    The public :func:`mm1d.reflectivity.stack_response` rejects negative p
    (no down-going wave leaving the surface toward -x), so the synthesis
    defines ``U(-p) := U(p)``; exact because every term of the recursion is a
    function of ``(p vp)^2``.
    """
    p = np.array([5.0e-5, 1.5e-4, 2.9e-4])
    terms_plus = _stack_terms(t2, p)
    terms_minus = _stack_terms(t2, -p)
    for plus, minus in zip(terms_plus, terms_minus, strict=True):
        np.testing.assert_allclose(plus, minus, atol=1e-15)


def test_offset_impulse_gather_event_times_match_synthetic(t1):
    """Impulse-domain synthesis places events at the same times as the Ricker one."""
    h, v = 500.0, 2000.0
    theta_test = np.array([10.0, 25.0])
    offsets = 2.0 * h * np.tan(np.radians(theta_test))
    times = 2.0 * h / (v * np.cos(np.radians(theta_test)))
    n_samples = round(1.5 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    impulse = offset_impulse_gather(
        t1, "primaries", ANGLES_1DEG, offsets, DT_S, n_samples, n_fft=n_fft
    )
    # the |w| filter sharpens spikes; find the largest |sample| in a window
    for t_theory, trace in zip(times, impulse, strict=True):
        i0 = round(float(t_theory) / DT_S)
        window = np.asarray(trace)[i0 - 15 : i0 + 15]
        peak_index = i0 - 15 + int(np.argmax(np.abs(window)))
        assert abs(peak_index * DT_S - t_theory) <= 1.5 * DT_S


# -- forward slant stack -----------------------------------------------------------


def test_slant_stack_hyperbola_to_intercept(t1):
    """Independent forward anchor: a pure hyperbolic gather transforms to tau(p)."""
    # constant-velocity hyperbola events (T1-like): tau(p) = T0 sqrt(1 - p^2 v^2)
    t0, v = 0.5, 2000.0
    offsets = np.linspace(0.0, 800.0, 81)
    t_s = np.arange(round(1.0 / DT_S)) * DT_S
    gather = np.zeros((offsets.size, t_s.size))
    n_wavelet = 151  # odd: symmetric about its center sample
    wavelet = np.asarray(
        ricker_wavelet((np.arange(n_wavelet, dtype=np.float64) - n_wavelet // 2) * DT_S, F_DOM_HZ)
    )
    for i, x in enumerate(offsets):
        arrival = np.sqrt(t0**2 + (x / v) ** 2)
        center = round(arrival / DT_S)
        lo = center - n_wavelet // 2
        gather[i, lo : lo + n_wavelet] += wavelet
    p_grid = np.atleast_1d(np.asarray(slowness_from_angle(np.array([0.0, 15.0, 30.0]), v)))
    transformed = slant_stack(gather, offsets, p_grid, DT_S, n_fft=FFT_OVERSAMPLE * t_s.size)
    tau_expected = t0 * np.sqrt(1.0 - (p_grid * v) ** 2)
    for tau, trace in zip(tau_expected, transformed, strict=True):
        pick = pick_in_window(envelope(trace), float(tau))
        assert abs(pick - tau) <= 1.5 * DT_S


@pytest.mark.parametrize(("model_name", "variant"), [("t1", "primaries"), ("t2", "full")])
def test_slant_stack_round_trip(canonical_models, model_name, variant):
    """Angle gather -> offset gather -> forward transform recovers it eventwise.

    Raw traces match in correlation and scale within finite-aperture
    corrections (the mirrored -p background of the exact bi-infinite pair
    does not cancel on one-sided grids); event *times* are exact.
    """
    model = canonical_models[model_name]
    n_samples = round({"t1": 1.5, "t2": 3.0}[model_name] / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    original = angle_gather(model, variant, ANGLES_1DEG, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft)
    offsets = default_offset_grid(model, ANGLES_1DEG)
    synthesized = offset_gather(
        model, variant, ANGLES_1DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    p_grid = np.atleast_1d(np.asarray(slowness_from_angle(ANGLES_1DEG, model.vp_m_s[0])))
    recovered = slant_stack(synthesized, offsets, p_grid, DT_S, n_fft=n_fft)
    angle_mask = ANGLES_1DEG <= 27.0  # away from the T2 mute (30 deg) and aperture edge
    time_mask = np.arange(n_samples) * DT_S <= 0.7 * n_samples * DT_S
    got = recovered[np.ix_(angle_mask, time_mask)]
    want = original[np.ix_(angle_mask, time_mask)]
    correlation = float(np.corrcoef(got.ravel(), want.ravel())[0, 1])
    scale = float(np.sum(got * want) / np.sum(want * want))
    assert correlation >= 0.94
    assert 0.85 <= scale <= 1.05
    for angle in (0.0, 10.0, 20.0):
        i = int(np.flatnonzero(angle == ANGLES_1DEG)[0])
        pick_got = pick_in_window(envelope(recovered[i]), float(np.argmax(np.abs(want[i]))) * DT_S)
        pick_want = pick_in_window(envelope(original[i]), float(np.argmax(np.abs(want[i]))) * DT_S)
        assert abs(pick_got - pick_want) <= 1.5 * DT_S


# -- validation ---------------------------------------------------------------------


def test_offset_gather_validation(t1):
    n_samples = round(0.5 / DT_S)
    with pytest.raises(ValueError, match="offsets_m must be 1-D"):
        offset_gather(t1, "primaries", ANGLES_1DEG, np.zeros((2, 2)), DT_S, n_samples, F_DOM_HZ)
    with pytest.raises(ValueError, match="non-finite"):
        offset_gather(
            t1, "primaries", ANGLES_1DEG, np.array([0.0, np.nan]), DT_S, n_samples, F_DOM_HZ
        )
    with pytest.raises(ValueError, match="at least one offset"):
        offset_gather(t1, "primaries", ANGLES_1DEG, np.array([]), DT_S, n_samples, F_DOM_HZ)
    with pytest.raises(ValueError, match="strictly increasing"):
        offset_gather(
            t1,
            "primaries",
            np.array([10.0, 5.0, 20.0]),
            np.array([0.0]),
            DT_S,
            n_samples,
            F_DOM_HZ,
        )
    with pytest.raises(ValueError, match="at least 2 angles"):
        offset_gather(t1, "primaries", np.array([10.0]), np.array([0.0]), DT_S, n_samples, F_DOM_HZ)
    with pytest.raises(ValueError, match="n_fft"):
        offset_gather(
            t1,
            "primaries",
            ANGLES_1DEG,
            np.array([0.0]),
            DT_S,
            n_samples,
            F_DOM_HZ,
            n_fft=n_samples - 1,
        )


def test_slant_stack_validation():
    traces = np.zeros((3, 50))
    offsets = np.array([0.0, 100.0, 200.0])
    with pytest.raises(ValueError, match="2-D"):
        slant_stack(np.zeros(50), offsets, np.array([0.0]), DT_S)
    with pytest.raises(ValueError, match="strictly increasing"):
        slant_stack(traces, np.array([0.0, 200.0, 100.0]), np.array([0.0]), DT_S)
    with pytest.raises(ValueError, match="entries but traces has"):
        slant_stack(traces, offsets[:2], np.array([0.0]), DT_S)
    with pytest.raises(ValueError, match="non-finite"):
        slant_stack(traces, offsets, np.array([0.0, np.inf]), DT_S)
    with pytest.raises(ValueError, match="non-empty 1-D"):
        slant_stack(traces, offsets, np.zeros((2, 2)), DT_S)
