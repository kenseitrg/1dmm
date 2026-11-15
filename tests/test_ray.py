"""Diagnostic gathers against analytic fractional Ricker events and recursion."""

import numpy as np
import pytest

from mm1d.events import PathLimitError, make_event, path_amplitude
from mm1d.gathers import path_offset_time
from mm1d.ray import event_moveout, ray_offset_gather, wavelet_tail_margin
from mm1d.reflectivity import zero_offset_trace
from mm1d.wavelet import ricker_wavelet


def test_t1_fractional_primary_and_multiple_hyperbolas(t1):
    offsets = np.array([0.0, 123.4, -500.0, 500.0])
    dt, n, f = 0.002, 850, 20.0
    result = ray_offset_gather(t1, "full", offsets, dt, n, f)
    expected = np.zeros_like(result.traces)
    t = np.arange(n) * dt
    for j in range(3):
        t0 = (j + 1) * 0.5
        arrival = np.sqrt(t0 * t0 + (offsets / 2000) ** 2)
        angle = np.degrees(np.arctan(np.abs(offsets) / ((j + 1) * 1000)))
        r = np.asarray(t1.reflection(0, angle)).real
        expected += (r * (-r) ** j)[:, None] * np.asarray(
            ricker_wavelet(t[None, :] - arrival[:, None], f)
        )
        event = result.catalog.events[j]
        assert event.vrms_m_s == pytest.approx(2000.0)
        np.testing.assert_allclose(result.arrival_times_s[j], arrival, atol=1e-13)
    np.testing.assert_allclose(result.traces, expected, atol=1e-14)
    assert np.all(result.supported)
    np.testing.assert_array_equal(result.traces[2], result.traces[3])
    solo = ray_offset_gather(t1, "full", offsets[1:2], dt, n, f)
    np.testing.assert_array_equal(solo.traces[0], result.traces[1])


def test_snell_geometry_independent_paths(t3):
    angles = np.array([0.0, 10.0, 20.0, 29.0])
    for points in ([-1, 3, -1], [-1, 2, 1, 2, -1], [-1, 0, -1, 2, -1]):
        event = make_event(t3, points)
        x, times = path_offset_time(t3, points, angles)
        p, got, support = event_moveout(t3, event, x)
        assert np.all(support)
        np.testing.assert_allclose(got, times, atol=2e-12)
        np.testing.assert_allclose(p, np.sin(np.radians(angles)) / 1800.0, atol=1e-16)


@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
@pytest.mark.parametrize("variant", ["full", "no_surface", "primaries"])
def test_normal_path_synthetic_independent_reflectivity(canonical_models, model_name, variant):
    model = canonical_models[model_name]
    n, dt = (2000 if model_name == "t3" else 1500), 0.002
    result = ray_offset_gather(model, variant, np.array([0.0]), dt, n, 20.0)
    assert "reflections" not in result.catalog.reached_bounds
    reference = zero_offset_trace(model, variant, dt, n, 20.0, n_fft=32 * n)
    # Includes wavelet-tail omission, padded FFT periodicity, and sampling error.
    np.testing.assert_allclose(result.traces[0], reference, atol=2e-9, rtol=0.0)


def test_t2_water_multiple_pinned_velocity(t2):
    result = ray_offset_gather(t2, "full", np.array([0.0, 250.0, 600.0]), 0.002, 1100, 20.0)
    for i, event in enumerate(result.catalog.events):
        if set(event.turning_points) == {-1, 0}:
            assert event.vrms_m_s == pytest.approx(1500.0)
            ok = result.supported[i]
            np.testing.assert_allclose(
                result.arrival_times_s[i, ok],
                np.sqrt(event.t0_s**2 + (np.array([0.0, 250.0, 600.0])[ok] / 1500.0) ** 2),
                atol=1e-12,
            )


def test_unsupported_not_clipped_and_validity_explicit(t2):
    result = ray_offset_gather(t2, "primaries", np.array([0.0, 500.0, 3000.0]), 0.002, 1500, 20.0)
    assert result.supported[0].tolist() == [True, False, False]
    assert np.isnan(result.arrival_times_s[0, 1])
    assert not result.valid_samples[1].all()
    assert result.valid_samples[0].all()
    assert np.all(result.traces[2] == 0.0)
    # Outside the aperture the primary is missing, not moved to a wrong time.
    time = np.sqrt(0.4**2 + (500 / 1500) ** 2)
    window = np.abs(np.arange(1500) * 0.002 - time) < 0.02
    assert not np.any(result.valid_samples[1, window])


def test_order_convergence_and_bounds(t3):
    results = [
        ray_offset_gather(
            t3, "full", np.array([0.0, 200.0, 600.0]), 0.004, 1000, 20.0, max_reflections=n
        )
        for n in (5, 11, 17, 23, 32)
    ]
    assert "reflections" in results[0].catalog.reached_bounds
    assert "reflections" not in results[-1].catalog.reached_bounds
    errors = [np.max(np.abs(r.traces - results[-1].traces)) for r in results[:-1]]
    assert errors[1] < errors[0]
    assert errors[2] < 1e-12
    assert errors[3] == 0.0
    with pytest.raises(PathLimitError):
        ray_offset_gather(t3, "full", np.array([0.0]), 0.004, 1000, 20.0, max_paths=5)


def test_event_free_windows_relative_to_weak_multiple(t1):
    # Exact analytic residual, normalized to a weak 2nd surface multiple, not
    # to the strong whole-gather peak. No visual clipping or arbitrary mute.
    offsets = np.array([0.0, 200.0, 600.0])
    result = ray_offset_gather(t1, "full", offsets, 0.002, 850, 20.0)
    t = np.arange(850) * 0.002
    weakest = t1.reflections_normal[0] ** 3
    for i, x in enumerate(offsets):
        arrivals = np.sqrt(np.array([0.5, 1.0, 1.5]) ** 2 + (x / 2000) ** 2)
        free = np.min(np.abs(t[:, None] - arrivals), axis=1) > 0.12
        # At 600 m all events are still supported; the analytic tails are tiny.
        assert np.max(np.abs(result.traces[i, free])) < 1e-12 * weakest


@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
@pytest.mark.parametrize("variant", ["full", "no_surface", "primaries"])
def test_all_models_event_free_windows(canonical_models, model_name, variant):
    model = canonical_models[model_name]
    result = ray_offset_gather(model, variant, np.array([0.0, 200.0, 600.0]), 0.002, 2000, 20.0)
    # Normalize to an independently specified weak target, not the global peak.
    points = [-1, 2, 1, 2, -1] if model_name == "t3" else [-1, 0, -1, 0, -1, 0, -1]
    weak_target = abs(path_amplitude(model, make_event(model, points)))
    times = np.arange(2000) * 0.002
    for i in range(3):
        arrivals = result.arrival_times_s[:, i]
        arrivals = arrivals[np.isfinite(arrivals)]
        distance = np.min(np.abs(times[:, None] - arrivals), axis=1)
        free = (distance > 0.12) & result.valid_samples[i]
        assert np.any(free & (times > model.interface_twt_s[0]))
        assert np.max(np.abs(result.traces[i, free])) < 1e-12 * weak_target


def test_empty_tail_and_bad_inputs(t1):
    result = ray_offset_gather(t1, "full", np.array([0.0]), 0.002, 50, 20.0)
    assert result.catalog.events == ()
    assert result.supported.shape == (0, 1)
    assert np.all(result.traces == 0.0)
    margin = wavelet_tail_margin(20.0)
    assert abs(float(ricker_wavelet(margin, 20.0))) == pytest.approx(1e-10, rel=1e-9)
    # Event just beyond record end still contributes its leading wavelet lobe.
    tail = ray_offset_gather(t1, "primaries", np.array([0.0]), 0.002, 246, 20.0)
    assert np.max(np.abs(tail.traces)) > 0.05
    with pytest.raises(ValueError):
        ray_offset_gather(t1, "full", np.array([0.0]), 0.002, 50, 20.0, max_angle_deg=90.0)
    with pytest.raises(ValueError):
        ray_offset_gather(t1, "full", np.array([0.0]), 0.002, 50, 20.0, wavelet_tail_tolerance=0.0)
