"""Ridge accounting is exact-support energy bookkeeping, not event picking."""

from dataclasses import replace

import numpy as np
import pytest

from mm1d.ray import ray_offset_gather
from mm1d.ridges import attribute_ridge_cells, strong_ridge_cells
from mm1d.semblance import SemblanceResult, semblance_with_fold


def test_gate_rms_and_strength_exclude_scale_free_tails():
    x = np.arange(4, dtype=float)
    traces = np.ones((4, 100))
    traces[:, 60:] *= 1e-9
    scan = semblance_with_fold(traces, x, 0.002, np.array([0.04, 0.16]), np.array([2000.0]), 0.01)
    np.testing.assert_allclose(scan.panel, 1.0)
    np.testing.assert_allclose(scan.gate_rms, [[1.0, 1e-9]])
    np.testing.assert_array_equal(strong_ridge_cells(scan), [[True, False]])
    # RMS is a pressure, semblance is scale-invariant.
    scaled = semblance_with_fold(
        3 * traces, x, 0.002, np.array([0.04, 0.16]), np.array([2000.0]), 0.01
    )
    np.testing.assert_allclose(scaled.gate_rms, 3 * scan.gate_rms)
    np.testing.assert_allclose(scaled.panel, scan.panel)
    with pytest.raises(ValueError, match="thresholds"):
        strong_ridge_cells(scan, relative_rms_min=0)
    with pytest.raises(ValueError, match="min_fold"):
        strong_ridge_cells(scan, min_fold=True)


def test_zero_gather_and_low_fold_are_not_strong():
    scan = SemblanceResult(np.ones((1, 2)), np.array([[1, 4]]), np.array([[1.0, 0.0]]))
    assert not strong_ridge_cells(scan).any()


@pytest.mark.parametrize("variant", ["full", "no_surface", "primaries"])
def test_attribution_exact_energy_and_scan_membership(t2, variant):
    x = np.linspace(0, 990, 31)
    ray = ray_offset_gather(t2, variant, x, 0.002, 1500, 20.0)
    t0 = np.linspace(0.25, 2.9, 41)
    v = np.linspace(1400, 2400, 13)
    scan = semblance_with_fold(
        ray.traces, x, 0.002, t0, v, 0.05, p_mute_s_m=ray.p_max_s_m, valid_samples=ray.valid_samples
    )
    selected = strong_ridge_cells(scan)
    result = attribute_ridge_cells(t2, ray, x, 0.002, 20.0, t0, v, selected, 0.05)
    iv, it = result.velocity_index, result.time_index
    assert iv.size > 0
    assert result.reconstruction_error < 1e-15
    # Independently connect scanner RMS to attributed sampled-gate energy.
    expected = scan.gate_rms[iv, it] ** 2 * scan.fold[iv, it] * 25
    np.testing.assert_allclose(result.observed_energy, expected, rtol=1e-14, atol=1e-20)
    assert np.all(result.group_energy >= 0)
    assert np.all(result.dominant_event_fold <= scan.fold[iv, it])
    assert np.all(np.isfinite(result.dominant_center_shift_s))
    if variant == "full":
        assert any(len(group) > 1 for group in result.groups)
        # Every enumerated ordering survives coherent grouping.
        assert sum(len(group) for group in result.groups) == len(ray.catalog.events)


def test_true_hyperbola_gate_shift_is_not_new_event(t1):
    x = np.linspace(0, 300, 21)
    ray = ray_offset_gather(t1, "primaries", x, 0.002, 600, 20.0)
    t0 = np.array([0.5, 0.53])
    v = np.array([2000.0])
    selected = np.ones((1, 2), dtype=bool)
    result = attribute_ridge_cells(t1, ray, x, 0.002, 20.0, t0, v, selected, 0.05)
    assert result.group_energy.shape == (1, 2)
    assert abs(result.dominant_center_shift_s[0]) < 1e-12
    assert result.dominant_center_shift_s[1] > 0.028
    assert result.dominant_differential_error_s[1] < 0.002
    np.testing.assert_allclose(result.group_energy[0], result.observed_energy)


def test_destructive_interference_is_not_unaccounted_energy(t3):
    x = np.linspace(0, 1000, 31)
    ray = ray_offset_gather(t3, "full", x, 0.002, 1100, 20.0)
    t0, v = np.array([1.78, 1.8, 1.82]), np.array([2400.0, 2500.0, 2600.0])
    result = attribute_ridge_cells(
        t3, ray, x, 0.002, 20.0, t0, v, np.ones((3, 3), dtype=bool), 0.05
    )
    assert result.reconstruction_error < 1e-15
    cross = result.observed_energy - result.group_energy.sum(axis=0)
    assert np.any(cross < -0.01 * result.observed_energy)
    # Families are variant-appropriate, not a surface marker on primaries.
    assert any({e.family for e in group} == {"mixed", "surface"} for group in result.groups)


def test_empty_selection_and_validation(t1):
    x = np.linspace(0, 300, 11)
    ray = ray_offset_gather(t1, "primaries", x, 0.002, 600, 20.0)
    args = (t1, ray, x, 0.002, 20.0, np.array([0.5]), np.array([2000.0]))
    result = attribute_ridge_cells(*args, np.zeros((1, 1), dtype=bool), 0.05)
    assert result.group_energy.shape == (1, 0)
    assert result.reconstruction_error < 1e-15
    empty = ray_offset_gather(t1, "primaries", x, 0.002, 50, 20.0)
    with pytest.raises(ValueError, match="empty event catalog"):
        attribute_ridge_cells(
            t1,
            empty,
            x,
            0.002,
            20.0,
            np.array([0.05]),
            np.array([2000.0]),
            np.ones((1, 1), dtype=bool),
            0.02,
        )
    with pytest.raises(ValueError, match="selected"):
        attribute_ridge_cells(*args, np.zeros((2, 2), dtype=bool), 0.05)
    truncated = replace(ray, catalog=replace(ray.catalog, reached_bounds=("reflections",)))
    with pytest.raises(ValueError, match="convergence"):
        attribute_ridge_cells(
            t1,
            truncated,
            x,
            0.002,
            20.0,
            np.array([0.5]),
            np.array([2000.0]),
            np.ones((1, 1), dtype=bool),
            0.05,
        )
