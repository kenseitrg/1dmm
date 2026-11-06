"""Zero-offset impulse responses and synthetics against analytic anchors (M2).

Anchor logic (PLAN.md T1/T2/T3, AGENTS.md sign conventions):

- T1 (single interface, layer TWT 0.5 s): ``full`` arrivals at
  ``(n+1) * TWT`` with amplitudes ``R1 * (R_s R1)^n`` = ``R1 * (-R1)^n``
  (alternating polarity, the free-surface ``R_s = -1`` signature);
  ``primaries`` and ``no_surface`` = single arrival ``R1`` (no interbed
  possible with one interface).
- T2 (water bottom): pure water-bottom train at 0.4(n+1) s = ``R1 (-R1)^n``
  (no transmission terms -- the path never crosses the water bottom); deep
  primary at 0.9 s = ``R2 * (1 - R1^2)`` (round-trip transmission through
  the water layer); first interbed at 1.4 s = ``-(1 - R1^2) R1 R2^2``.
- Records are longer than the plan display defaults when 1e-6-level amplitude
  fidelity is asserted: FFT wraparound folds the (dense, exponentially
  decaying) multiple tail beyond the record back into it (measured: the
  fold-back level drops below 1e-9 at T >= 8 s for T1, 24 s for T2, 32 s
  for T3).
"""

import numpy as np
import pytest

from mm1d.reflectivity import (
    ALL_VARIANTS,
    FREE_SURFACE_REFLECTIVITY,
    impulse_response,
    stack_response_normal,
    variant_multiples,
    zero_offset_trace,
)
from mm1d.wavelet import ricker_wavelet

DT_S = 2.0e-3
"""Default sample interval (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Dominant Ricker frequency (PLAN.md numerical defaults)."""


def i_s(time_s: float) -> int:
    """Sample index of an on-grid arrival time at ``DT_S`` sampling."""
    index = round(time_s / DT_S)
    assert abs(index * DT_S - time_s) < 1e-12, f"arrival {time_s} s is off the sample grid"
    return index


# -- variant switches and conventions ------------------------------------------


def test_variant_multiples_table():
    assert variant_multiples("full") == (True, True)
    assert variant_multiples("no_surface") == (False, True)
    assert variant_multiples("primaries") == (False, False)


def test_variant_multiples_rejects_unknown():
    with pytest.raises(ValueError, match="unknown wavefield variant"):
        variant_multiples("everything")  # ty: ignore[invalid-argument-type]


def test_free_surface_convention_is_pressure_release():
    # AGENTS.md: R_s = -1 (with +1 the multiple train would be all-positive;
    # the alternating T1/T2 trains below are the physical regression guard)
    assert FREE_SURFACE_REFLECTIVITY == -1.0


def test_response_is_hermitian(t2):
    freq = np.linspace(-50.0, 50.0, 401, dtype=np.float64)
    response = stack_response_normal(t2, "full", freq)
    np.testing.assert_allclose(response, np.conj(response[::-1]), atol=1e-15, rtol=0.0)


# -- T1: single-interface analytic anchor ---------------------------------------


def test_t1_full_multiple_train(t1):
    layer_twt_s = float(t1.interface_twt_s[0])
    assert layer_twt_s == pytest.approx(0.5, abs=1e-12)
    r1 = float(t1.reflections_normal[0])
    n_samples = 4000  # T = 8 s
    trace = impulse_response(t1, "full", DT_S, n_samples)
    for n in range(6):
        assert trace[i_s((n + 1) * layer_twt_s)] == pytest.approx(r1 * (-r1) ** n, abs=1e-6), (
            f"arrival {n + 1}"
        )
    # nothing off the arrival comb (arrivals every layer TWT; the arrival at
    # exactly T aliases onto sample 0)
    residual = np.abs(trace)
    for n in range(int(8.0 / layer_twt_s)):
        residual[i_s((n + 1) * layer_twt_s) % n_samples] = 0.0
    assert residual.max() < 1e-10


def test_t1_primaries_single_arrival(t1):
    r1 = float(t1.reflections_normal[0])
    trace = impulse_response(t1, "primaries", DT_S, 4000)
    assert trace[i_s(0.5)] == pytest.approx(r1, abs=1e-10)
    assert np.delete(np.abs(trace), i_s(0.5)).max() < 1e-12


def test_t1_no_surface_equals_primaries(t1):
    # one interface -> no interbed possible -> the two variants coincide
    full_path = impulse_response(t1, "no_surface", DT_S, 4000)
    prim = impulse_response(t1, "primaries", DT_S, 4000)
    np.testing.assert_allclose(full_path, prim, atol=1e-12)


# -- T2: water-bottom model ------------------------------------------------------


@pytest.fixture(scope="module")
def t2_variants(t2):
    """Impulse responses of all three variants on a long record (T = 24 s)."""
    n_samples = 12_000
    return t2, {v: impulse_response(t2, v, DT_S, n_samples) for v in ALL_VARIANTS}


def test_t2_water_bottom_train(t2_variants):
    t2, traces = t2_variants
    r1 = float(t2.reflections_normal[0])
    # 0.4..2.0 s are the clean pure-water-bottom times; from 2.4 s on, the
    # sediment-reverberation family (0.4*1 + 0.5*4 = 2.4 s and beyond)
    # coincides with the train and the simple series no longer holds
    for n in range(5):
        assert traces["full"][i_s(0.4 * (n + 1))] == pytest.approx(r1 * (-r1) ** n, abs=1e-8), (
            f"water-bottom multiple order {n}"
        )


def test_t2_deep_primary_with_transmissions(t2_variants):
    t2, traces = t2_variants
    r1, r2 = t2.reflections_normal
    t_down, t_up = t2.transmissions_normal
    assert traces["full"][i_s(0.9)] == pytest.approx(r2 * t_down[0] * t_up[0], abs=1e-8)
    # round-trip transmission identity T_down * T_up = 1 - R^2
    assert t_down[0] * t_up[0] == pytest.approx(1.0 - r1**2, rel=1e-12)
    # primaries and no_surface carry the same primary, unaffected by R_s
    assert traces["primaries"][i_s(0.9)] == pytest.approx(r2 * (1.0 - r1**2), abs=1e-8)
    assert traces["no_surface"][i_s(0.9)] == pytest.approx(r2 * (1.0 - r1**2), abs=1e-8)


def test_t2_no_surface_has_no_surface_multiples(t2_variants):
    _, traces = t2_variants
    # pure water-bottom train times carry only surface-multiple energy
    for time_s in (0.8, 1.2, 1.6, 2.0):
        assert abs(traces["no_surface"][i_s(time_s)]) < 1e-10, f"at {time_s} s"


def test_t2_interbed_first_order_amplitude(t2_variants):
    t2, traces = t2_variants
    r1, r2 = t2.reflections_normal
    interbed = traces["no_surface"] - traces["primaries"]
    # deep primary + one extra reverberation in the sediment layer (0.5 s TWT):
    # path T0_down * R2 * (-R1) * R2 * T0_up at 0.9 s + 0.5 s
    expected = -(1.0 - r1**2) * r1 * r2**2
    assert interbed[i_s(1.4)] == pytest.approx(expected, rel=1e-6)


def test_t2_variant_differences_vanish_before_first_return(t2_variants):
    _, traces = t2_variants
    pre = slice(0, i_s(0.35))  # first return at 0.4 s
    assert np.abs((traces["full"] - traces["no_surface"])[pre]).max() < 1e-9
    assert np.abs((traces["no_surface"] - traces["primaries"])[pre]).max() < 1e-12


# -- T3: interbed demonstrator ----------------------------------------------------


@pytest.fixture(scope="module")
def t3_variants(t3):
    """Impulse responses of all three variants on a long record (T = 32 s)."""
    n_samples = 16_000
    return t3, {v: impulse_response(t3, v, DT_S, n_samples) for v in ALL_VARIANTS}


def test_t3_variant_differences_vanish_before_first_return(t3_variants):
    _, traces = t3_variants
    pre = slice(0, i_s(0.45))  # first return at 0.5 s
    assert np.abs((traces["full"] - traces["no_surface"])[pre]).max() < 1e-9
    assert np.abs((traces["no_surface"] - traces["primaries"])[pre]).max() < 1e-10


def test_t3_surface_difference_first_surface_multiple(t3_variants):
    t3, traces = t3_variants
    r1 = float(t3.reflections_normal[0])
    surface_only = traces["full"] - traces["no_surface"]
    # first surface multiple of the first primary: 2 * 0.5 s, amplitude -R1^2
    assert surface_only[i_s(1.0)] == pytest.approx(-(r1**2), abs=1e-6)


def test_t3_interbed_difference_first_interbed(t3_variants):
    _, traces = t3_variants
    interbed = traces["no_surface"] - traces["primaries"]
    # primary from interface 2 at 1.3 s; first internal reverberation inside
    # the soft interbed adds its TWT 0.3 s -> 1.6 s
    assert abs(interbed[i_s(1.6)]) > 1e-4
    # ...and the interbed series continues at 0.3 s period
    assert abs(interbed[i_s(1.9)]) > 1e-5


# -- synthetics: wavelet convolution ----------------------------------------------


def test_t2_synthetic_peak_equals_impulse_amplitude(t2):
    r1 = float(t2.reflections_normal[0])
    trace = zero_offset_trace(t2, "full", DT_S, 12_000, F_DOM_HZ)
    # isolated event: wavelet peak (=1) times impulse amplitude, exactly
    assert trace[i_s(0.4)] == pytest.approx(r1, rel=1e-6)
    assert trace[i_s(0.8)] == pytest.approx(-(r1**2), rel=1e-6)
    assert trace[i_s(1.2)] == pytest.approx(r1**3, rel=1e-5)


def test_zero_offset_trace_equals_time_domain_convolution(t2):
    n_samples = 12_000
    impulse = impulse_response(t2, "full", DT_S, n_samples)
    trace = zero_offset_trace(t2, "full", DT_S, n_samples, F_DOM_HZ)
    half = 300  # +/- 0.6 s wavelet window: Ricker tails are zero well inside
    wavelet = np.asarray(
        ricker_wavelet(np.arange(-half, half + 1, dtype=np.float64) * DT_S, F_DOM_HZ)
    )
    convolved = np.convolve(impulse, wavelet)[half : half + n_samples]
    np.testing.assert_allclose(trace, convolved, atol=1e-9)


def test_variant_differences_vanish_before_first_return_synthetic(t2):
    # display-default window (T = 3 s) on an extended FFT record: no wraparound
    n_samples = 1500
    n_fft = 12_000
    traces = {
        v: zero_offset_trace(t2, v, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft) for v in ALL_VARIANTS
    }
    pre = slice(0, i_s(0.35))
    for earlier, later in (("no_surface", "full"), ("primaries", "no_surface")):
        assert np.abs((traces[later] - traces[earlier])[pre]).max() < 1e-9


def test_n_fft_extends_periodicity(t2):
    # short display window, long FFT record: equals the head of the long record
    short = impulse_response(t2, "full", DT_S, 1500, n_fft=12_000)
    long_head = impulse_response(t2, "full", DT_S, 12_000)[:1500]
    np.testing.assert_allclose(short, long_head, atol=1e-15)
    # without n_fft the same short record contains wrapped-in late multiples
    wrapped = impulse_response(t2, "full", DT_S, 1500)
    assert np.abs(wrapped[: i_s(0.35)]).max() > 1e-3


# -- input validation ---------------------------------------------------------------


def test_impulse_response_input_validation(t1):
    with pytest.raises(ValueError, match="dt_s"):
        impulse_response(t1, "full", 0.0, 1000)
    with pytest.raises(ValueError, match="dt_s"):
        impulse_response(t1, "full", -2.0e-3, 1000)
    with pytest.raises(ValueError, match="n_samples"):
        impulse_response(t1, "full", DT_S, 1)
    with pytest.raises(ValueError, match="n_fft"):
        impulse_response(t1, "full", DT_S, 1000, n_fft=999)
    with pytest.raises(ValueError, match="unknown wavefield variant"):
        impulse_response(t1, "soup", DT_S, 1000)  # ty: ignore[invalid-argument-type]


def test_zero_offset_trace_input_validation(t1):
    with pytest.raises(ValueError, match="dt_s"):
        zero_offset_trace(t1, "full", 0.0, 1000, F_DOM_HZ)
    with pytest.raises(ValueError, match="f_dom_hz"):
        zero_offset_trace(t1, "full", DT_S, 1000, 0.0)
