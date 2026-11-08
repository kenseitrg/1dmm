"""Angle-gather physics validation against analytic anchors (M3).

Anchor logic (PLAN.md M3 acceptance + AGENTS.md conventions):

- ``stack_response`` at ``p = 0`` must reproduce the M2 normal-incidence path
  exactly (the omega-p recursion is its strict generalization).
- T1 at any subcritical angle has exact closed forms: ``primaries`` is
  ``R(p) e^{-2 i w tau}`` with intercept time ``tau = 2 h cos(theta)/v``;
  ``full`` is the surface-multiple series ``x / (1 + x)``, ``x = R e^{-2 i w tau}``
  (water-bottom-style train with amplitudes ``R (R_s R)^n`` at angle).
- T2 ``primaries``/``no_surface`` have exact two-interface closed forms at
  angle (Snell-refracted intercept times per layer, products of angle-dependent
  coefficients); past the half-space critical angle (theta > 30 deg in water)
  the coefficients are complex and the engine convention
  (:func:`mm1d.reflectivity._stack_terms`) is the conjugate of the
  ``e^{-i w t}``-convention coefficients of :mod:`mm1d.model`.
- ``full`` equals the free-surface series of ``no_surface``:
  ``S_full = S_ns / (1 - R_s S_ns)``.
- Post-critical slowness (``p >= 1/v_max``) makes the stack a lossless total
  reflector (``|S_ns| = 1``) and the ``full`` surface series resonant;
  gathers mute those traces while the recursion itself stays finite and
  warning-free across the whole grid.
"""

import numpy as np
import pytest
from scipy.signal import hilbert

from mm1d.gathers import (
    angle_gather,
    angle_impulse_gather,
    critical_mute_slowness,
    primary_intercept_times,
    slowness_from_angle,
)
from mm1d.model import LayeredModel, acoustic_reflection_p, acoustic_transmission_p
from mm1d.reflectivity import (
    ALL_VARIANTS,
    _stack_terms,
    impulse_response,
    stack_response,
    stack_response_normal,
    zero_offset_trace,
)

DT_S = 2.0e-3
"""Default sample interval (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Dominant Ricker frequency (PLAN.md numerical defaults)."""

ANGLES_1DEG = np.arange(0.0, 40.0 + 1.0e-9, 1.0)
"""Default gather-angle grid from PLAN.md (0-40 deg, 1 deg step)."""

FREQ = np.linspace(0.0, 200.0, 401)
"""Frequency grid for exact (time-grid-independent) closed-form comparisons."""


def conj_coefficient(value) -> complex:
    """Engine-convention coefficient: conjugate of the model (e^{-i w t}) one."""
    return np.conj(complex(value))


def exp_delay(freq_hz: np.ndarray, delay_s: float) -> np.ndarray:
    """Delay factor ``exp(-i w t0)`` of the engine (irfft) convention."""
    return np.exp(-2j * np.pi * freq_hz * delay_s)


# -- engine reduces to the M2 normal-incidence path at p = 0 --------------------


@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_stack_response_matches_normal_at_zero_slowness(canonical_models, model_name, variant):
    model = canonical_models[model_name]
    got = stack_response(model, variant, FREQ, 0.0)
    expected = stack_response_normal(model, variant, FREQ)
    np.testing.assert_allclose(got, expected, atol=1e-12)


# -- T1: exact closed forms at angle ---------------------------------------------


@pytest.mark.parametrize("theta_deg", [0.0, 7.0, 13.0, 25.0, 40.0])
def test_t1_primaries_closed_form_vs_angle(t1, theta_deg):
    v0 = float(t1.vp_m_s[0])
    tau = 2.0 * float(t1.h_m[0]) * np.cos(np.radians(theta_deg)) / v0
    r1 = conj_coefficient(t1.reflection(0, theta_deg))
    expected = r1 * exp_delay(FREQ, tau)
    got = stack_response(t1, "primaries", FREQ, slowness_from_angle(theta_deg, v0))
    np.testing.assert_allclose(got, expected, atol=1e-12)
    # one interface: no interbed possible -> no_surface coincides with primaries
    got_ns = stack_response(t1, "no_surface", FREQ, slowness_from_angle(theta_deg, v0))
    np.testing.assert_allclose(got_ns, expected, atol=1e-12)


@pytest.mark.parametrize("theta_deg", [0.0, 7.0, 13.0, 25.0, 40.0])
def test_t1_full_surface_series_vs_angle(t1, theta_deg):
    v0 = float(t1.vp_m_s[0])
    tau = 2.0 * float(t1.h_m[0]) * np.cos(np.radians(theta_deg)) / v0
    r1 = conj_coefficient(t1.reflection(0, theta_deg))
    x = r1 * exp_delay(FREQ, tau)
    expected = x / (1.0 + x)  # U0 / (1 - R_s U0) with R_s = -1
    got = stack_response(t1, "full", FREQ, slowness_from_angle(theta_deg, v0))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_t1_amplitude_trend_grows_toward_critical(t1):
    # acoustic R(theta) for a velocity/impedance increase grows toward +1
    amplitudes = [abs(complex(t1.reflection(0, theta))) for theta in (0.0, 20.0, 40.0)]
    assert amplitudes == sorted(amplitudes)
    assert amplitudes[0] == pytest.approx(0.304, abs=2e-3)


# -- T2: exact two-interface closed forms at angle --------------------------------


def t2_primaries_closed_form(t2, theta_deg: float) -> np.ndarray:
    """Single-scattering response of T2 at angle: two primary terms."""
    v_water, v_sed = float(t2.vp_m_s[0]), float(t2.vp_m_s[1])
    h_water, h_sed = float(t2.h_m[0]), float(t2.h_m[1])
    theta_rad = np.radians(theta_deg)
    # Snell refraction into the sediment (propagating for theta < 38.68 deg)
    sin_sed = np.sin(theta_rad) * v_sed / v_water
    assert sin_sed <= 1.0, "sediment leg evanescent: primaries have no real arrival"
    theta_sed = np.arcsin(sin_sed)
    tau_water = 2.0 * h_water * np.cos(theta_rad) / v_water
    tau_sed = 2.0 * h_sed * np.cos(theta_sed) / v_sed
    r_wb, (td_wb, tu_wb) = (
        conj_coefficient(t2.reflection(0, theta_deg)),
        tuple(conj_coefficient(c) for c in t2.transmission(0, theta_deg)),
    )
    r_hs = conj_coefficient(t2.reflection(1, np.degrees(theta_sed)))
    return r_wb * exp_delay(FREQ, tau_water) + (
        td_wb * tu_wb * r_hs * exp_delay(FREQ, tau_water + tau_sed)
    )


@pytest.mark.parametrize(
    "theta_deg",
    # 29 deg instead of 30: exactly at the half-space critical angle R'(p) -> inf
    # and ulp-level slowness differences amplify to ~1e-8 in R(p) there
    [0.0, 5.0, 10.0, 20.0, 29.0, 35.0],
)
def test_t2_primaries_closed_form_vs_angle(t2, theta_deg):
    got = stack_response(t2, "primaries", FREQ, slowness_from_angle(theta_deg, float(t2.vp_m_s[0])))
    np.testing.assert_allclose(got, t2_primaries_closed_form(t2, theta_deg), atol=1e-12)


@pytest.mark.parametrize("theta_deg", [0.0, 5.0, 10.0, 20.0, 29.0, 35.0])
def test_t2_no_surface_closed_form_vs_angle(t2, theta_deg):
    v_water, v_sed = float(t2.vp_m_s[0]), float(t2.vp_m_s[1])
    h_water, h_sed = float(t2.h_m[0]), float(t2.h_m[1])
    theta_rad = np.radians(theta_deg)
    theta_sed = np.arcsin(np.sin(theta_rad) * v_sed / v_water)
    tau_water = 2.0 * h_water * np.cos(theta_rad) / v_water
    tau_sed = 2.0 * h_sed * np.cos(theta_sed) / v_sed
    r_wb = conj_coefficient(t2.reflection(0, theta_deg))
    td_wb, tu_wb = (conj_coefficient(c) for c in t2.transmission(0, theta_deg))
    r_hs = conj_coefficient(t2.reflection(1, np.degrees(theta_sed)))
    # compound sub-stack below interface 0 with the interbed denominator
    x = r_hs * exp_delay(FREQ, tau_sed)
    g0 = r_wb + td_wb * tu_wb * x / (1.0 + r_wb * x)
    expected = exp_delay(FREQ, tau_water) * g0
    got = stack_response(
        t2, "no_surface", FREQ, slowness_from_angle(theta_deg, float(t2.vp_m_s[0]))
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_full_equals_surface_series_of_no_surface(t3):
    # S_full = S_ns / (1 - R_s S_ns) with R_s = -1, at every (theta, f)
    angles = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    p = np.atleast_1d(np.asarray(slowness_from_angle(angles, float(t3.vp_m_s[0]))))
    s_ns = stack_response(t3, "no_surface", FREQ, p)
    s_full = stack_response(t3, "full", FREQ, p)
    np.testing.assert_allclose(s_full, s_ns / (1.0 + s_ns), rtol=1e-9, atol=1e-12)


# -- energy / stability / Hermiticity ---------------------------------------------


@pytest.mark.parametrize("model_name", ["t2", "t3"])
def test_no_surface_bounded_by_energy_conservation(canonical_models, model_name):
    # lossless stack: compound reflection magnitude <= 1 at every subcritical p
    # (exactly 1 once any interface is post-critical: total reflection)
    model = canonical_models[model_name]
    p = np.atleast_1d(np.asarray(slowness_from_angle(ANGLES_1DEG, float(model.vp_m_s[0]))))
    assert np.max(np.abs(stack_response(model, "no_surface", FREQ, p))) <= 1.0 + 1e-9


@pytest.mark.parametrize("theta_deg", [10.0, 20.0, 28.0])
def test_response_is_hermitian_vs_frequency(t2, theta_deg):
    freq = np.linspace(-200.0, 200.0, 801)
    p = slowness_from_angle(theta_deg, float(t2.vp_m_s[0]))
    got = stack_response(t2, "full", freq, p)
    np.testing.assert_allclose(got, np.conj(got[::-1]), atol=1e-12)


@pytest.mark.parametrize("model_name", ["t2", "t3"])
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_recursion_stable_across_full_p_grid(canonical_models, model_name, variant):
    # no overflow/invalid warnings at (or past) critical slowness; the full
    # variant's surface-series poles stay finite on a generic frequency grid
    model = canonical_models[model_name]
    p = np.atleast_1d(np.asarray(slowness_from_angle(ANGLES_1DEG, float(model.vp_m_s[0]))))
    with np.errstate(over="raise", invalid="raise"):
        response = stack_response(model, variant, FREQ, p)
    assert np.isfinite(response).all()


# -- gathers ------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["t2", "t3"])
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_theta0_column_equals_zero_offset_trace(canonical_models, model_name, variant):
    model = canonical_models[model_name]
    n_samples, n_fft = 1500, 12_000
    gather = angle_impulse_gather(
        model, variant, np.array([0.0, 20.0]), DT_S, n_samples, n_fft=n_fft
    )
    reference = impulse_response(model, variant, DT_S, n_samples, n_fft=n_fft)
    np.testing.assert_allclose(gather[0], reference, atol=1e-10)


@pytest.mark.parametrize("model_name", ["t2", "t3"])
def test_theta0_synthetic_column_equals_zero_offset_trace(canonical_models, model_name):
    model = canonical_models[model_name]
    n_samples, n_fft = 1500, 12_000
    gather = angle_gather(
        model, "full", np.array([0.0, 20.0]), DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    reference = zero_offset_trace(model, "full", DT_S, n_samples, F_DOM_HZ, n_fft=n_fft)
    np.testing.assert_allclose(gather[0], reference, atol=1e-10)


@pytest.mark.parametrize("theta_deg", [0.0, 15.0, 30.0, 40.0])
def test_t1_first_arrival_envelope_time_and_amplitude(t1, theta_deg):
    # isolated first arrival: envelope peak at 0.5 cos(theta) s with amplitude
    # R1(theta) (wavelet peak = 1; envelope is rotation-invariant, so this
    # also holds for complex post-critical coefficients)
    dt = 1.0e-3
    n_samples = 1500
    gather = angle_gather(
        t1, "full", np.array([theta_deg]), dt, n_samples, F_DOM_HZ, n_fft=4 * n_samples
    )
    envelope = np.abs(hilbert(gather[0]))
    i_peak = int(np.argmax(envelope))
    assert i_peak * dt == pytest.approx(0.5 * np.cos(np.radians(theta_deg)), abs=2 * dt)
    assert envelope[i_peak] == pytest.approx(abs(complex(t1.reflection(0, theta_deg))), rel=2e-3)


def test_postcritical_mute(t2):
    angles = np.array([28.0, 29.0, 30.0, 31.0, 40.0])
    gather = angle_gather(t2, "full", angles, DT_S, 1500, F_DOM_HZ, n_fft=6000)
    for row, theta in zip(gather, angles, strict=True):
        if theta >= 30.0:  # p >= 1/v_max (half-space): muted, all variants
            assert np.abs(row).max() == 0.0, f"theta = {theta}"
        else:
            assert np.abs(row).max() > 1e-3, f"theta = {theta}"
    # the engine itself still computes (finite) responses past the mute
    p_mute = critical_mute_slowness(t2)
    assert p_mute is not None
    assert p_mute == pytest.approx(1.0 / 3000.0)
    assert np.isfinite(stack_response(t2, "full", FREQ, 1.05 * p_mute)).all()


def test_no_mute_when_no_critical_angle():
    model = LayeredModel(
        h_m=np.array([100.0]),
        vp_m_s=np.array([2000.0, 1500.0]),
        rho_kg_m3=np.array([2000.0, 2100.0]),
    )
    assert critical_mute_slowness(model) is None
    gather = angle_gather(model, "full", ANGLES_1DEG, DT_S, 1500, F_DOM_HZ, n_fft=6000)
    assert np.abs(gather).max() > 0.0


# -- plane-ray intercept times -------------------------------------------------------


def test_t2_primary_intercept_times(t2):
    tau = primary_intercept_times(t2, np.array([0.0, 30.0, 39.0]))
    np.testing.assert_allclose(tau[0], np.array([0.4, 0.9]), atol=1e-12)
    # Snell refraction: sediment angle asin(sin(30 deg) * 2400/1500) = 53.13 deg
    theta_sed = np.arcsin(0.5 * 2400.0 / 1500.0)
    expected = [0.4 * np.cos(np.radians(30.0)), 0.0]
    expected[1] = expected[0] + 2.0 * 600.0 * np.cos(theta_sed) / 2400.0
    np.testing.assert_allclose(tau[1], expected, atol=1e-12)
    # beyond the sediment critical angle the deep primary has no real arrival
    assert np.isfinite(tau[2, 0])
    assert np.isnan(tau[2, 1])


def test_t3_primary_intercept_times(t3):
    tau = primary_intercept_times(t3, np.array([0.0]))
    np.testing.assert_allclose(tau[0], np.array([0.5, 1.0, 1.3, 1.8]), atol=1e-12)
    # binding critical angles per event (fastest layer on the path, in
    # surface-layer terms): P1/P2/P3 -> asin(1800/2700) = 41.8 deg (whole grid),
    # P4 -> asin(1800/3000) = 36.9 deg (fastest finite layer is 3000 m/s)
    tau = primary_intercept_times(t3, ANGLES_1DEG)
    assert np.isfinite(tau[:, :3]).all()
    finite_p4 = np.isfinite(tau[:, 3])
    assert finite_p4[ANGLES_1DEG <= 36.0].all()
    assert not finite_p4[ANGLES_1DEG >= 37.0].any()
    # intercept times decrease with angle (plane-wave moveout in tau-p)
    assert np.all(np.diff(tau[:, :3], axis=0) < 0.0)


def test_slowness_from_angle():
    assert slowness_from_angle(0.0, 1500.0) == 0.0
    assert slowness_from_angle(30.0, 1500.0) == pytest.approx(0.5 / 1500.0, rel=1e-12)
    with pytest.raises(ValueError, match="vp_m_s"):
        slowness_from_angle(30.0, 0.0)


# -- shapes and input validation -------------------------------------------------------


def test_shapes(t2):
    assert stack_response(t2, "full", FREQ, 0.0).shape == FREQ.shape
    p = np.atleast_1d(np.asarray(slowness_from_angle(np.array([0.0, 10.0, 20.0]), 1500.0)))
    assert stack_response(t2, "full", FREQ, p).shape == (3, FREQ.size)
    gather = angle_impulse_gather(t2, "full", np.array([0.0, 10.0]), DT_S, 500, n_fft=2000)
    assert gather.shape == (2, 500)


def test_stack_response_input_validation(t1, t2):
    p_limit = 1.0 / float(t1.vp_m_s[0])
    with pytest.raises(ValueError, match="slowness"):
        stack_response(t1, "full", FREQ, p_limit)
    with pytest.raises(ValueError, match="slowness"):
        stack_response(t1, "full", FREQ, -0.1)
    with pytest.raises(ValueError, match="1-D"):
        stack_response(t1, "full", FREQ, np.full((2, 2), 1.0e-4))
    with pytest.raises(ValueError, match="1-D"):
        stack_response(t1, "full", np.stack([FREQ, FREQ]), 0.0)
    with pytest.raises(ValueError, match="unknown wavefield variant"):
        stack_response(t1, "soup", FREQ, 0.0)  # ty: ignore[invalid-argument-type]
    # angles just below grazing are fine (propagating in the surface layer)
    assert np.isfinite(stack_response(t2, "full", FREQ, 0.999 * p_limit)).all()


def test_gather_input_validation(t1):
    with pytest.raises(ValueError, match="angle"):
        angle_gather(t1, "full", np.array([90.0]), DT_S, 500, F_DOM_HZ)
    with pytest.raises(ValueError, match="angle"):
        angle_gather(t1, "full", np.array([-1.0]), DT_S, 500, F_DOM_HZ)
    with pytest.raises(ValueError, match="non-finite"):
        angle_gather(t1, "full", np.array([np.nan]), DT_S, 500, F_DOM_HZ)
    with pytest.raises(ValueError, match="at least one angle"):
        angle_gather(t1, "full", np.array([]), DT_S, 500, F_DOM_HZ)
    with pytest.raises(ValueError, match="dt_s"):
        angle_impulse_gather(t1, "full", np.array([0.0]), 0.0, 500)
    with pytest.raises(ValueError, match="n_fft"):
        angle_impulse_gather(t1, "full", np.array([0.0]), DT_S, 500, n_fft=499)
    with pytest.raises(ValueError, match="unknown wavefield variant"):
        angle_impulse_gather(t1, "soup", np.array([0.0]), DT_S, 500)  # ty: ignore[invalid-argument-type]


def test_engine_coefficients_conjugate_model_api(t3):
    # _stack_terms is the conj of model.acoustic_*_p wherever the latter is
    # defined (p <= 1/vp_above per interface): identical while real
    # (subcritical), conjugated past the critical angle
    p = np.atleast_1d(np.asarray(slowness_from_angle(ANGLES_1DEG, float(t3.vp_m_s[0]))))
    r, t_down, t_up, two_way = _stack_terms(t3, p)
    assert two_way.shape == (t3.n_interfaces, p.size)
    for i in range(t3.n_interfaces):
        valid = p <= 1.0 / float(t3.vp_m_s[i])
        r_m = np.asarray(
            acoustic_reflection_p(
                float(t3.vp_m_s[i]),
                float(t3.rho_kg_m3[i]),
                float(t3.vp_m_s[i + 1]),
                float(t3.rho_kg_m3[i + 1]),
                p[valid],
            )
        )
        td_m, tu_m = acoustic_transmission_p(
            float(t3.vp_m_s[i]),
            float(t3.rho_kg_m3[i]),
            float(t3.vp_m_s[i + 1]),
            float(t3.rho_kg_m3[i + 1]),
            p[valid],
        )
        np.testing.assert_allclose(r[i, valid], np.conj(r_m), atol=1e-15)
        np.testing.assert_allclose(t_down[i, valid], np.conj(np.asarray(td_m)), atol=1e-15)
        np.testing.assert_allclose(t_up[i, valid], np.conj(np.asarray(tu_m)), atol=1e-15)
