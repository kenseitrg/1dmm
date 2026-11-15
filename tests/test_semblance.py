"""Semblance physics validation against analytic anchors (M5).

Anchor logic (PLAN.md M5 acceptance, corrected expectation 2026-11-09, +
the scan-design notes in ``mm1d.semblance``):

- **Neidell-Taner panel** (:func:`mm1d.semblance.semblance_panel`): by
  Cauchy-Schwarz ``0 <= S <= 1``; identical traces -> ``S = 1``; incoherent
  noise -> ``S ~ 1/N``.  Hand-built hyperbola events are picked at their true
  ``(T0, V)`` cells; record/slowness exclusions zero out unconstrained cells;
  scale-free coherence ties near 1 across the wavelet extent on noise-free
  gathers (why scan gathers carry a seeded 1 % incoherent floor).
- **Dix anchors on engine gathers** (T1-T3, M4 synthesis + noise floor +
  taper-clean slowness bound): deep-interface primaries pin to Dix RMS
  velocities within about one velocity cell; the *shallowest* interface of
  each model is aperture-limited (``x_max/T0 ~ 0.7`` at the taper-clean
  bound, ~20 Hz wavelet) and resolves its stacking velocity only to
  ~+/-100 m/s -- the resolution floor asserted instead of the one-cell
  tolerance.
- **Free-surface multiple signature (the M5 payoff)**: the T2 ``full`` panel
  shows the water-bottom multiple train (orders 1-4) as a ridge pinned at the
  water velocity at every order -- *below* the same-T0 Dix primary trend --
  and the peg-leg ``peg12`` ridge between the pin and that Dix trend.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from mm1d.gathers import offset_gather
from mm1d.semblance import (
    default_t0_grid,
    default_velocity_grid,
    dix_rms_velocities,
    scan_slowness_bound,
    semblance_panel,
)

DT_S = 2.0e-3
"""Default sample interval (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Dominant Ricker frequency (PLAN.md numerical defaults)."""

ANGLES_1DEG = np.arange(0.0, 40.0 + 1.0e-9, 1.0)
"""Default synthesis aperture (PLAN.md numerical defaults)."""

FFT_OVERSAMPLE = 8
"""FFT record extension factor (M2-M4 practice)."""

GATE_S = 0.05
"""Semblance gate: one dominant period of the Ricker source, s."""

SPREAD_CAP_M_PER_S = 1100.0
"""Scan aperture velocity: max offset = SPREAD_CAP * T0(deepest primary), m."""

NOISE_FLOOR_FRACTION = 0.01
"""Seeded incoherent noise floor of scan gathers, fraction of peak."""

NOISE_SEED = 20261110
"""Seed of the noise floor (deterministic anchors, same recipe as the script)."""

RECORDS_S: dict[str, float] = {
    "single_interface": 1.5,
    "water_bottom": 3.0,
    "thick_5layer": 4.0,
}
"""Display record length per canonical model, s."""


# -- shared scan recipe (mirrors scripts/semblance.py) -------------------------


def scan_offsets(model, n_offsets: int = 61) -> NDArray[np.float64]:
    """Uniform scan offsets of the M5 demo recipe, m."""
    x_max = SPREAD_CAP_M_PER_S * float(model.interface_twt_s[-1])
    return np.linspace(0.0, x_max, n_offsets)


def build_panel(model, variant, record_s: float) -> tuple[NDArray, NDArray, NDArray]:
    """Build one scan panel with the demo recipe (noise floor + slowness bound)."""
    n_samples = round(record_s / DT_S)
    offsets = scan_offsets(model)
    gather = offset_gather(
        model,
        variant,
        ANGLES_1DEG,
        offsets,
        DT_S,
        n_samples,
        F_DOM_HZ,
        n_fft=FFT_OVERSAMPLE * n_samples,
    )
    rng = np.random.default_rng(NOISE_SEED)
    sigma = NOISE_FLOOR_FRACTION * float(np.abs(gather).max())
    gather = gather + rng.normal(0.0, sigma, gather.shape)
    t0_grid = default_t0_grid(record_s)
    v_grid = default_velocity_grid(model)
    panel = semblance_panel(
        gather,
        offsets,
        DT_S,
        t0_grid,
        v_grid,
        GATE_S,
        p_mute_s_m=scan_slowness_bound(model, ANGLES_1DEG),
    )
    return t0_grid, v_grid, panel


def window_pick(
    panel: NDArray[np.float64],
    t0_grid: NDArray[np.float64],
    v_grid: NDArray[np.float64],
    t0_s: float,
    v_m_s: float,
    *,
    half_t0_s: float = 0.05,
    half_v_m_s: float = 100.0,
) -> tuple[float, float, float]:
    """Argmax of the panel within a window around ``(t0_s, v_m_s)``."""
    t_mask = np.abs(t0_grid - t0_s) <= half_t0_s
    v_mask = np.abs(v_grid - v_m_s) <= half_v_m_s
    window = panel[np.ix_(v_mask, t_mask)]
    i_v, i_t = np.unravel_index(int(np.argmax(window)), window.shape)
    v_indices = np.flatnonzero(v_mask)
    t_indices = np.flatnonzero(t_mask)
    return (
        float(t0_grid[t_indices[i_t]]),
        float(v_grid[v_indices[i_v]]),
        float(window[i_v, i_t]),
    )


def gaussian_gather(
    t0_s: float,
    v_m_s: float,
    offsets_m: NDArray[np.float64],
    dt_s: float,
    n_samples: int,
    sigma_s: float = 0.012,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Analytic gather: one Gaussian wavelet on an exact hyperbola."""
    t_s = np.arange(n_samples, dtype=np.float64) * dt_s
    pick = np.sqrt(t0_s**2 + (offsets_m / v_m_s) ** 2)
    return amplitude * np.exp(-0.5 * ((t_s[np.newaxis, :] - pick[:, np.newaxis]) / sigma_s) ** 2)


def dix_trend_v(model, t0_s: float) -> float:
    """Dix RMS velocity of a hypothetical primary at ``t0_s``, m/s (constant-v layers)."""
    twt = model.layer_twt_s
    vp = model.vp_m_s[:-1]
    k = min(int(np.searchsorted(np.cumsum(twt), t0_s, side="right")), twt.size - 1)
    t_above = float(np.sum(twt[:k]))
    sum_tv2 = float(np.sum(twt[:k] * vp[:k] ** 2)) + (t0_s - t_above) * vp[k] ** 2
    return float(np.sqrt(sum_tv2 / t0_s))


# -- module fixtures: engine panels built once ---------------------------------


@pytest.fixture(scope="module")
def t2_primaries_panel(t2):
    """(t0 grid, v grid, panel) of the T2 primaries variant."""
    return build_panel(t2, "primaries", RECORDS_S["water_bottom"])


@pytest.fixture(scope="module")
def t2_full_panel(t2):
    """(t0 grid, v grid, panel) of the T2 full variant."""
    return build_panel(t2, "full", RECORDS_S["water_bottom"])


@pytest.fixture(scope="module")
def t1_full_panel(t1):
    """(t0 grid, v grid, panel) of the T1 full variant."""
    return build_panel(t1, "full", RECORDS_S["single_interface"])


@pytest.fixture(scope="module")
def t3_primaries_panel(t3):
    """(t0 grid, v grid, panel) of the T3 primaries variant."""
    return build_panel(t3, "primaries", RECORDS_S["thick_5layer"])


# -- pure panel function: hand-built anchors -----------------------------------


def test_noise_gather_incoherent_floor():
    rng = np.random.default_rng(3)
    gather = rng.standard_normal((41, 1000))
    offsets = np.linspace(0.0, 1000.0, 41)
    t0_grid = default_t0_grid(2.0, n_t0=101)
    v_grid = np.linspace(1500.0, 2500.0, 41)
    panel = semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S)
    assert panel.shape == (41, 101)
    assert panel.mean() == pytest.approx(1.0 / 41.0, abs=0.005)
    assert panel.max() < 0.15


def test_hyperbola_event_picked_at_true_cell():
    offsets = np.linspace(0.0, 1000.0, 61)
    t0_grid = default_t0_grid(1.5, n_t0=151)
    v_grid = np.linspace(1500.0, 2500.0, 41)
    gather = gaussian_gather(0.5, 2000.0, offsets, DT_S, 750)
    panel = semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S)
    i_v, i_t = np.unravel_index(int(np.argmax(panel)), panel.shape)
    assert abs(t0_grid[i_t] - 0.5) <= 1.5 * (t0_grid[1] - t0_grid[0])
    assert abs(v_grid[i_v] - 2000.0) <= 1.5 * (v_grid[1] - v_grid[0])
    assert panel.max() >= 0.95


def test_two_events_separately_picked():
    # carries the documented noise floor: without it, a uniform gate shift on
    # the wavelet flank is a scale-free S -> 1 tie and the argmax is arbitrary
    rng = np.random.default_rng(5)
    offsets = np.linspace(0.0, 1000.0, 61)
    t0_grid = default_t0_grid(2.0, n_t0=201)
    v_grid = np.linspace(1200.0, 2600.0, 71)
    gather = gaussian_gather(0.4, 1500.0, offsets, DT_S, 1000) + 0.4 * gaussian_gather(
        0.9, 2050.0, offsets, DT_S, 1000
    )
    gather = gather + rng.normal(0.0, 0.01 * float(np.abs(gather).max()), gather.shape)
    panel = semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S)
    for t0, v, expected_s in ((0.4, 1500.0, 0.9), (0.9, 2050.0, 0.9)):
        p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, t0, v)
        assert abs(p_t0 - t0) <= 1.5 * (t0_grid[1] - t0_grid[0])
        assert abs(p_v - v) <= 1.5 * (v_grid[1] - v_grid[0])
        assert p_s >= expected_s


def test_panel_values_bounded():
    rng = np.random.default_rng(11)
    offsets = np.linspace(0.0, 800.0, 41)
    t0_grid = default_t0_grid(2.0, n_t0=101)
    v_grid = np.linspace(1400.0, 2400.0, 31)
    gather = gaussian_gather(0.6, 1800.0, offsets, DT_S, 1000) + 0.05 * rng.standard_normal(
        (41, 1000)
    )
    panel = semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S)
    assert np.all(panel >= 0.0)
    assert np.all(panel <= 1.0 + 1.0e-9)


def test_record_exclusion_and_min_traces():
    # hyperbola so steep that most traces' gates leave the record near its end:
    # T0 = 1.4 s, V = 700 m/s, record 1.5 s -> gate in record only for
    # x <= 700 * sqrt(1.5^2 - 1.4^2) ~ 375 m of the 1000 m spread
    offsets = np.linspace(0.0, 1000.0, 61)
    t0_grid = default_t0_grid(1.5, n_t0=151)
    v_grid = np.array([700.0])
    gather = gaussian_gather(1.4, 700.0, offsets, DT_S, 750)
    it = np.argmin(np.abs(t0_grid - 1.4))
    # ~23 of 61 traces valid (< 0.5 but > 0.2): unconstrained at the default 0.5
    panel = semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S, min_traces_fraction=0.5)
    assert panel[0, it] == 0.0
    # with the small default fraction the surviving near offsets still stack
    panel = semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S, min_traces_fraction=0.2)
    assert panel[0, it] > 0.8


def test_slowness_bound_excludes_offsets():
    offsets = np.linspace(0.0, 1000.0, 61)
    t0_grid = np.array([0.5])
    v_grid = np.array([2000.0])
    gather = gaussian_gather(0.5, 2000.0, offsets, DT_S, 750)
    p_local = lambda x: x / (2000.0**2 * np.sqrt(0.25 + (x / 2000.0) ** 2))  # noqa: E731
    # bound at x = 400 m: ~25 of 61 traces stay valid (> 0.2 fraction) and the
    # event still stacks coherently on its valid subset
    panel = semblance_panel(
        gather, offsets, DT_S, t0_grid, v_grid, GATE_S, p_mute_s_m=p_local(400.0)
    )
    assert panel[0, 0] > 0.8
    # bound at x = 150 m: ~10 of 61 valid (< 0.2) -> unconstrained cell
    panel = semblance_panel(
        gather, offsets, DT_S, t0_grid, v_grid, GATE_S, p_mute_s_m=p_local(150.0)
    )
    assert panel[0, 0] == 0.0


# -- scan grids and helpers ------------------------------------------------------


def test_default_t0_grid_covers_record():
    grid = default_t0_grid(3.0, n_t0=301)
    assert grid[0] == 0.0
    assert grid.size == 301
    assert np.all(np.diff(grid) > 0.0)
    assert grid[-1] == pytest.approx(3.0 - 3.0 / 301)
    with pytest.raises(ValueError, match="record_s"):
        default_t0_grid(0.0)
    with pytest.raises(ValueError, match="n_t0"):
        default_t0_grid(3.0, n_t0=1)


def test_default_velocity_grid_bounds(t2):
    v_grid = default_velocity_grid(t2, n_v=81)
    assert v_grid.size == 81
    assert np.all(np.diff(v_grid) > 0.0)
    assert v_grid[0] == pytest.approx(0.8 * 1500.0)
    assert v_grid[-1] == pytest.approx(1.25 * float(np.max(dix_rms_velocities(t2))))
    # the grid must cover the Dix anchors and the pinned water velocity
    assert v_grid[0] < 1500.0 < v_grid[-1]
    assert v_grid[-1] > float(np.max(dix_rms_velocities(t2)))
    with pytest.raises(ValueError, match="n_v"):
        default_velocity_grid(t2, n_v=1)
    with pytest.raises(ValueError, match="low_factor"):
        default_velocity_grid(t2, low_factor=0.0)
    with pytest.raises(ValueError, match="high_factor"):
        default_velocity_grid(t2, high_factor=0.5)


def test_dix_rms_velocities_closed_form(t2):
    # computed from the raw layer table, independent of the helper's internals
    twt = np.array([2.0 * 300.0 / 1500.0, 2.0 * 600.0 / 2400.0])
    vp = np.array([1500.0, 2400.0])
    expected = np.sqrt(np.cumsum(twt * vp**2) / np.cumsum(twt))
    np.testing.assert_allclose(dix_rms_velocities(t2), expected, rtol=1e-12)


def test_scan_slowness_bound_taper_clean(t1, t2):
    # T2: the post-critical mute (1/3000) binds before the 40 deg aperture edge
    assert scan_slowness_bound(t2, ANGLES_1DEG) == pytest.approx(0.85 / 3000.0, rel=1e-12)
    # T1: no mute below the aperture edge (critical at 41.8 deg) -> 0.85 x p(40 deg)
    p_edge = np.sin(np.radians(40.0)) / 2000.0
    assert scan_slowness_bound(t1, ANGLES_1DEG) == pytest.approx(0.85 * p_edge, rel=1e-12)
    with pytest.raises(ValueError, match="taper_fraction"):
        scan_slowness_bound(t2, ANGLES_1DEG, taper_fraction=1.0)


# -- engine anchors: T2 primaries vs Dix (M5 acceptance) ------------------------


def test_t2_primaries_deep_interface_at_dix(t2, t2_primaries_panel):
    t0_grid, v_grid, panel = t2_primaries_panel
    v_cell = v_grid[1] - v_grid[0]
    t_cell = t0_grid[1] - t0_grid[0]
    dix = dix_rms_velocities(t2)
    # P2 spans the full taper-clean aperture -> pinned to Dix within ~one cell
    p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, 0.9, float(dix[1]))
    assert abs(p_t0 - 0.9) <= 1.5 * t_cell
    assert abs(p_v - float(dix[1])) <= 1.5 * v_cell
    assert p_s >= 0.9


def test_t2_primaries_shallow_interface_resolution_floor(t2, t2_primaries_panel):
    t0_grid, v_grid, panel = t2_primaries_panel
    dix = dix_rms_velocities(t2)
    # P1 is limited to x <= ~280 m by the taper-clean bound (x/T0 ~ 0.7):
    # its velocity is resolved only to ~+/-100 m/s at 20 Hz (see module notes)
    p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, 0.4, float(dix[0]))
    assert abs(p_t0 - 0.4) <= 0.03
    assert abs(p_v - float(dix[0])) <= 100.0
    assert p_s >= 0.9


def test_t2_full_water_bottom_train_pinned(t2, t2_full_panel):
    t0_grid, v_grid, panel = t2_full_panel
    v_water = 1500.0
    for order in range(1, 5):
        t0 = (order + 1) * 0.4
        p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, t0, v_water)
        # pinned at the water velocity at every order (within ~1.5 cells)
        assert abs(p_v - v_water) <= 60.0, f"order {order}: pick {p_v:.1f} m/s"
        # ... clearly below the same-T0 Dix primary trend (radon separability)
        assert p_v < dix_trend_v(t2, t0) - 250.0
        # S stays high along the train; order 4 dips toward 0.85 where peg-legs
        # and sediment reverberations approach coincidence (M2 notes)
        assert p_s >= 0.8
        assert abs(p_t0 - t0) <= 0.03


def test_t2_full_peg_leg_between_pin_and_dix(t2, t2_full_panel):
    t0_grid, v_grid, panel = t2_full_panel
    # peg12: 4 water legs + 2 sediment legs -> T0 = 1.3 s,
    # V_stk = sqrt((0.8*1500^2 + 0.5*2400^2)/1.3) = 1897.4 m/s
    v_peg = float(np.sqrt((0.8 * 1500.0**2 + 0.5 * 2400.0**2) / 1.3))
    assert v_peg == pytest.approx(1897.4, abs=0.1)
    p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, 1.3, v_peg)
    v_cell = v_grid[1] - v_grid[0]
    assert abs(p_v - v_peg) <= 2.0 * v_cell
    assert abs(p_t0 - 1.3) <= 0.03
    assert p_s >= 0.9
    # between the pinned surface-multiple ridge and the Dix trend
    assert 1500.0 < p_v < dix_trend_v(t2, 1.3)


def test_t1_primary_and_multiple_pinned(t1, t1_full_panel):
    t0_grid, v_grid, panel = t1_full_panel
    v_cell = v_grid[1] - v_grid[0]
    # surface multiple M1 (pure layer-2 path, exact hyperbola): pinned at v1
    p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, 1.0, 2000.0)
    assert abs(p_t0 - 1.0) <= 0.015
    assert abs(p_v - 2000.0) <= 1.5 * v_cell
    assert p_s >= 0.9
    # shallow primary P1: resolution floor (x/T0 ~ 1.1, wavelet-broad)
    p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, 0.5, 2000.0)
    assert abs(p_t0 - 0.5) <= 0.04
    assert abs(p_v - 2000.0) <= 100.0
    assert p_s >= 0.9


def test_t3_primaries_at_dix(t3, t3_primaries_panel):
    t0_grid, v_grid, panel = t3_primaries_panel
    v_cell = v_grid[1] - v_grid[0]
    t_cell = t0_grid[1] - t0_grid[0]
    dix = dix_rms_velocities(t3)
    for k, (t0, v_dix) in enumerate(zip(t3.interface_twt_s, dix, strict=True)):
        p_t0, p_v, p_s = window_pick(panel, t0_grid, v_grid, float(t0), float(v_dix))
        if k == 0:
            # shallowest interface: resolution floor only
            assert abs(p_t0 - float(t0)) <= 0.04
            assert abs(p_v - float(v_dix)) <= 100.0
        else:
            assert abs(p_t0 - float(t0)) <= 1.5 * t_cell
            assert abs(p_v - float(v_dix)) <= 1.5 * v_cell
        assert p_s >= 0.9


# -- validation errors ------------------------------------------------------------


def test_semblance_panel_validation_errors():
    gather = np.zeros((11, 100))
    offsets = np.linspace(0.0, 500.0, 11)
    t0_grid = default_t0_grid(0.2, n_t0=21)
    v_grid = np.linspace(1500.0, 2500.0, 11)
    with pytest.raises(ValueError, match="gather must be 2-D"):
        semblance_panel(np.zeros(100), offsets, DT_S, t0_grid, v_grid, GATE_S)
    with pytest.raises(ValueError, match="at least 2 offset traces"):
        semblance_panel(np.zeros((1, 100)), np.array([0.0]), DT_S, t0_grid, v_grid, GATE_S)
    with pytest.raises(ValueError, match="non-finite"):
        bad = gather.copy()
        bad[0, 0] = np.nan
        semblance_panel(bad, offsets, DT_S, t0_grid, v_grid, GATE_S)
    with pytest.raises(ValueError, match="offsets_m has 5 entries"):
        semblance_panel(gather, np.linspace(0.0, 500.0, 5), DT_S, t0_grid, v_grid, GATE_S)
    with pytest.raises(ValueError, match="dt_s"):
        semblance_panel(gather, offsets, 0.0, t0_grid, v_grid, GATE_S)
    with pytest.raises(ValueError, match="gate_s"):
        semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, -0.05)
    with pytest.raises(ValueError, match="min_traces_fraction"):
        semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S, min_traces_fraction=0.0)
    with pytest.raises(ValueError, match="p_mute_s_m"):
        semblance_panel(gather, offsets, DT_S, t0_grid, v_grid, GATE_S, p_mute_s_m=-1.0)
    with pytest.raises(ValueError, match="t0_grid_s must be strictly increasing"):
        semblance_panel(gather, offsets, DT_S, t0_grid[::-1].copy(), v_grid, GATE_S)
    with pytest.raises(ValueError, match="v_grid_m_s"):
        semblance_panel(gather, offsets, DT_S, t0_grid, np.full(11, 2000.0), GATE_S)
