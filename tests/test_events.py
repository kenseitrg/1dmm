"""Independent path signs, multiplicity, and omega-p acceptance anchors."""

from typing import cast

import numpy as np
import pytest

from mm1d.events import (
    PathLimitError,
    display_events,
    enumerate_events,
    make_event,
    path_amplitude,
    path_amplitudes,
)
from mm1d.reflectivity import WavefieldVariant, stack_response


def test_single_interface_train(t1):
    r = t1.reflections_normal[0]
    catalog = enumerate_events(t1, "full", 3.0, max_reflections=20)
    assert len(catalog.events) == 6
    for n, event in enumerate(catalog.events):
        assert event.t0_s == pytest.approx((n + 1) * 0.5)
        assert path_amplitude(t1, event) == pytest.approx(r * (-r) ** n)
        assert event.vrms_m_s == pytest.approx(2000.0)
        assert event.surface_order == n
        assert event.internal_order == 0
    for variant in ("primaries", "no_surface"):
        assert len(enumerate_events(t1, variant, 3.0).events) == 1


@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
def test_primary_coefficients(canonical_models, model_name):
    model = canonical_models[model_name]
    events = enumerate_events(model, "primaries", 4.0).events
    r = model.reflections_normal
    for k, event in enumerate(events):
        assert event.t0_s == pytest.approx(model.interface_twt_s[k])
        assert path_amplitude(model, event) == pytest.approx(r[k] * np.prod(1 - r[:k] ** 2))
        assert event.turning_points == (-1, k, -1)


def test_hand_paths_underside_and_coincident_orderings(t2, t3):
    r = t3.reflections_normal
    internal = make_event(t3, [-1, 2, 1, 2, -1])
    assert internal.family == "interbed"
    assert internal.t0_s == pytest.approx(1.6)
    expected = (1 - r[0] ** 2) * (1 - r[1] ** 2) * r[2] ** 2 * (-r[1])
    assert path_amplitude(t3, internal) == pytest.approx(expected)
    mixed = make_event(t3, [-1, 2, 1, 2, -1, 0, -1])
    assert mixed.family == "mixed"
    assert path_amplitude(t3, mixed) == pytest.approx(-expected * r[0])
    a = make_event(t2, [-1, 1, -1, 0, -1])
    b = make_event(t2, [-1, 0, -1, 1, -1])
    assert a.traversal_counts == b.traversal_counts
    assert a.turning_points != b.turning_points
    cat = enumerate_events(t2, "full", 1.4)
    assert a in cat.events and b in cat.events
    assert path_amplitude(t2, a) == pytest.approx(
        -t2.reflections_normal[0] * t2.reflections_normal[1] * (1 - t2.reflections_normal[0] ** 2)
    )


@pytest.mark.parametrize("variant", ["full", "no_surface", "primaries"])
@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
def test_independent_path_spectrum(canonical_models, model_name, variant):
    model = canonical_models[model_name]
    frequencies = np.array([0.0, 3.3, 11.7, 24.0])
    errors = []
    orders = (3, 7, 17) if model_name == "t2" and variant == "full" else (3, 5, 7)
    for order in orders:
        cat = enumerate_events(model, variant, 30.0, max_reflections=order)
        error = 0.0
        for p in (0.0, 1.0e-4):
            q = np.sqrt(1 / model.vp_m_s[:-1] ** 2 - p * p)
            summed = np.zeros(frequencies.size, dtype=complex)
            for event in cat.events:
                tau = np.sum(np.asarray(event.traversal_counts) * model.h_m * q)
                summed += path_amplitude(model, event, p) * np.exp(-2j * np.pi * frequencies * tau)
            reference = stack_response(model, variant, frequencies, np.array([p]))[0]
            error = max(error, float(np.abs(summed - reference).max()))
        errors.append(error)
    if variant == "primaries":
        assert max(errors) < 2e-14
    else:
        assert errors[-1] < errors[0] + 1e-14
        # Finite-order frequency sums approach, but do not equal, the infinite
        # response. High-order late paths matter here, unlike a time window.
        assert errors[-1] < (0.03 if variant == "full" else 3e-3)
        if model_name != "t1" or variant == "full":
            assert errors[-1] < 0.6 * errors[0]


@pytest.mark.parametrize("variant", ["full", "no_surface", "primaries"])
@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
def test_subcritical_path_time_window(canonical_models, model_name, variant):
    from mm1d.gathers import angle_gather
    from mm1d.wavelet import ricker_wavelet

    model = canonical_models[model_name]
    p = 1.5e-4
    dt, n = 0.002, 1100
    times = np.arange(n) * dt
    c_min = np.sqrt(1 - (p * np.max(model.vp_m_s[:-1])) ** 2)
    cat = enumerate_events(model, variant, (times[-1] + 0.12) / c_min)
    assert "reflections" not in cat.reached_bounds
    q = np.sqrt(1 / model.vp_m_s[:-1] ** 2 - p * p)
    amplitudes = path_amplitudes(model, cat.events, np.array([p]))[:, 0]
    summed = np.zeros(n)
    for amplitude, event in zip(amplitudes, cat.events, strict=True):
        assert amplitude == pytest.approx(path_amplitude(model, event, p), abs=1e-15)
        tau = np.sum(np.array(event.traversal_counts) * model.h_m * q)
        summed += amplitude * np.asarray(ricker_wavelet(times - tau, 20.0))
    angle = np.degrees(np.arcsin(p * model.vp_m_s[0]))
    reference = angle_gather(model, variant, np.array([angle]), dt, n, 20.0, n_fft=32 * n)[0]
    np.testing.assert_allclose(summed, reference, atol=2e-9, rtol=0.0)


def test_catalog_bounds_and_display(t2):
    cat = enumerate_events(t2, "full", 3.0, max_reflections=2)
    assert "reflections" in cat.reached_bounds
    assert all(e.total_reflections <= 2 for e in cat.events)
    assert all(e.t0_s <= 3.0 + 1e-12 for e in cat.events)
    assert len(display_events(t2, cat, max_order=0)) == 2
    assert display_events(t2, cat, min_amplitude=2.0) == ()
    with pytest.raises(PathLimitError, match="max_paths"):
        enumerate_events(t2, "full", 4.0, max_paths=3)
    assert enumerate_events(t2, "full", 0.1).events == ()
    with pytest.raises(ValueError, match="integer"):
        enumerate_events(t2, "full", 3.0, max_reflections=0)
    with pytest.raises(ValueError, match="integer"):
        enumerate_events(t2, "full", 3.0, max_paths=cast("int", 1.5))
    with pytest.raises(ValueError, match="max_time"):
        enumerate_events(t2, "full", np.nan)
    with pytest.raises(ValueError, match="variant"):
        enumerate_events(t2, cast("WavefieldVariant", "bad"), 3.0)


@pytest.mark.parametrize(
    "points", [[-1, 0, 1, -1], [-1, 0, 0, -1], [0, 1, -1], [-1, 4, -1], [-1, 0.5, -1]]
)
def test_event_validation(t2, points):
    with pytest.raises(ValueError):
        make_event(t2, points)


@pytest.mark.parametrize("p", [-1.0, np.nan, 1 / 3000])
def test_coefficient_requires_global_subcritical(t2, p):
    with pytest.raises(ValueError, match="subcritical"):
        path_amplitude(t2, make_event(t2, [-1, 0, -1]), p)
