"""Independent moveout/resolution checks, not panel-argmax regression tests."""

import numpy as np
import pytest

from mm1d.events import make_event
from mm1d.gathers import path_offset_time
from mm1d.interpretation import (
    finite_spread_trajectory,
    fit_stacking_velocity,
    moveout_resolution,
)
from mm1d.ray import event_moveout


@pytest.mark.parametrize("t0,v", [(0.5, 2000.0), (0.8, 1500.0), (2.0, 1500.0)])
def test_exact_hyperbola(t0, v):
    x = np.linspace(-600, 600, 61)
    t = np.sqrt(t0**2 + (x / v) ** 2)
    fit = fit_stacking_velocity(x, t)
    assert fit.t0_s == pytest.approx(t0, abs=1e-12)
    assert fit.velocity_m_s == pytest.approx(v, abs=1e-7)
    assert fit.max_residual_s < 1e-12
    assert fit.fold == x.size


@pytest.mark.parametrize("order", [0, 1, 3])
def test_t2_water_train_geometry_and_support(t2, order):
    event = make_event(t2, [-1, *([0, -1] * (order + 1))])
    x = np.linspace(0, 1200, 121)
    fits = finite_spread_trajectory(t2, event, x, np.array([200.0, 600.0, 1200.0]))
    for fit in fits:
        assert fit is not None
        assert fit.t0_s == pytest.approx(0.4 * (order + 1), abs=1e-11)
        assert fit.velocity_m_s == pytest.approx(1500, abs=1e-6)
        assert fit.max_residual_s < 1e-11
    p_max = (1 - 1e-6) / 3000.0
    # Independent single-water-layer range at the global subcritical bound.
    x_limit = 2 * 300 * (order + 1) * p_max / np.sqrt(1 / 1500**2 - p_max**2)
    assert fits[-1] is not None
    assert fits[-1].fold == np.count_nonzero(x <= x_limit)
    assert fits[-1].max_offset_m <= x_limit


def test_layered_fit_independent_parametric_rays(t2):
    event = make_event(t2, [-1, 1, -1])
    p = np.linspace(0, 0.00030, 61)
    x, t = path_offset_time(t2, event.turning_points, np.rad2deg(np.arcsin(p * 1500.0)))
    fit = fit_stacking_velocity(x, t)
    assert fit.velocity_m_s > event.vrms_m_s + 5
    assert fit.max_residual_s > 1e-5
    # Direct formula independent of the bracketed inverse event_moveout.
    c = np.sqrt(1 - (p[:, None] * np.array([1500.0, 2400.0])) ** 2)
    expected_x = np.sum(
        2 * np.array([300.0, 600.0]) * p[:, None] * np.array([1500.0, 2400.0]) / c, axis=1
    )
    expected_t = np.sum(2 * np.array([300.0, 600.0]) / (np.array([1500.0, 2400.0]) * c), axis=1)
    np.testing.assert_allclose(x, expected_x, atol=1e-10)
    np.testing.assert_allclose(t, expected_t, atol=1e-14)
    near = fit_stacking_velocity(x[:5], t[:5])
    assert abs(near.velocity_m_s - event.vrms_m_s) < 1
    trajectory = finite_spread_trajectory(t2, event, x, np.array([x[4], x[-1] + 1]))
    assert trajectory[-1] is not None
    assert trajectory[-1].velocity_m_s == pytest.approx(fit.velocity_m_s, abs=1e-5)


def test_usable_mask_and_insufficient_offsets(t1):
    event = make_event(t1, [-1, 0, -1])
    x = np.arange(0, 601, 100.0)
    mask = np.array([True, True, True, False, False, False, False])
    fits = finite_spread_trajectory(t1, event, x, np.array([50.0, 100.0, 600.0]), usable=mask)
    assert fits[:2] == (None, None)
    assert fits[2] is not None
    assert fits[2].fold == 3
    assert fits[2].max_offset_m == 200
    with pytest.raises(ValueError, match="usable"):
        finite_spread_trajectory(
            t1, event, x, np.array([600.0]), usable=np.ones((2, 2), dtype=bool)
        )
    with pytest.raises(ValueError, match="spread limits"):
        finite_spread_trajectory(t1, event, x, np.array([600.0, 500.0]))


def test_resolution_is_joint_exact_timing_budget():
    x = np.linspace(0, 600, 31)
    t = np.sqrt(0.5**2 + (x / 2000) ** 2)
    v = np.arange(1500.0, 2601.0, 10.0)
    epsilon = 0.005
    envelope = moveout_resolution(x, t, v, epsilon)
    assert envelope.feasible[np.flatnonzero(v == 2000)[0]]
    assert not envelope.feasible[0]
    assert np.all(np.isnan(envelope.lower_t0_s[~envelope.feasible]))
    for j in np.flatnonzero(envelope.feasible):
        for intercept in (envelope.lower_t0_s[j], envelope.upper_t0_s[j]):
            assert np.max(np.abs(np.sqrt(intercept**2 + (x / v[j]) ** 2) - t)) <= epsilon + 1e-14
    tighter = moveout_resolution(x, t, v, epsilon / 10)
    assert tighter.feasible.sum() < envelope.feasible.sum()
    # A narrow supplied velocity range can censor BOTH uncertainty bounds.
    censored = moveout_resolution(x, t, np.array([1999.0, 2001.0]), epsilon)
    assert censored.feasible.all()


def test_nonhyperbolicity_can_exhaust_resolution_budget(t2):
    event = make_event(t2, [-1, 1, -1])
    x = np.linspace(0, 1200, 61)
    _, t, ok = event_moveout(t2, event, x)
    fit = fit_stacking_velocity(x[ok], t[ok])
    grid = np.linspace(fit.velocity_m_s - 100, fit.velocity_m_s + 100, 201)
    envelope = moveout_resolution(x[ok], t[ok], grid, 1e-7)
    assert not envelope.feasible.any()


@pytest.mark.parametrize(
    "x,t",
    [([0, 1], [1, 2]), ([0, 0, 1], [1, 1, 2]), ([0, 1, 2], [1, np.nan, 2]), ([0, 1, 2], [0, 1, 2])],
)
def test_invalid_arrivals(x, t):
    with pytest.raises(ValueError, match="distinct absolute offsets"):
        fit_stacking_velocity(np.array(x), np.array(t))


def test_invalid_resolution_and_degenerate_fit():
    x = np.arange(3.0)
    t = np.sqrt(1 + x * x)
    for v in (np.array([2.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan, 2.0])):
        with pytest.raises(ValueError, match="velocity grid"):
            moveout_resolution(x, t, v, 0.01)
    with pytest.raises(ValueError, match="tolerance"):
        moveout_resolution(x, t, np.array([1.0, 2.0]), 0.0)
    with pytest.raises(ValueError, match="positive hyperbola"):
        fit_stacking_velocity(x, np.array([3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="positive hyperbola"):
        fit_stacking_velocity(x, np.ones(3))
