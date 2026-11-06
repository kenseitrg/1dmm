"""Ricker wavelet: analytic properties and time/frequency consistency."""

import numpy as np
import pytest

from mm1d.wavelet import ricker_spectrum, ricker_wavelet

F_DOM_HZ = 20.0
"""Dominant frequency of the canonical v1 source (PLAN.md default)."""


def test_unit_peak_and_symmetry():
    t_s = np.linspace(-0.25, 0.25, 2001, dtype=np.float64)
    w = np.asarray(ricker_wavelet(t_s, F_DOM_HZ))
    assert w[1000] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(w, w[::-1], atol=1e-15)


def test_side_lobe_minima():
    # dw/dt = 0 at (pi f_dom t)^2 = 3/2, value -2 exp(-3/2)
    t_min_s = np.sqrt(1.5) / (np.pi * F_DOM_HZ)
    value = ricker_wavelet(t_min_s, F_DOM_HZ)
    assert value == pytest.approx(-2.0 * np.exp(-1.5), rel=1e-12)
    assert value == pytest.approx(-0.44626, abs=1e-4)


def test_zero_mean():
    # W(0) = 0: the time-domain integral vanishes
    t_s = np.linspace(-2.0, 2.0, 400_001, dtype=np.float64)
    w = ricker_wavelet(t_s, F_DOM_HZ)
    assert np.trapezoid(w, t_s) == pytest.approx(0.0, abs=1e-14)


def test_spectrum_closed_form_values():
    f_hz = np.array([0.0, F_DOM_HZ, 2.0 * F_DOM_HZ])
    w_spec = np.asarray(ricker_spectrum(f_hz, F_DOM_HZ))
    assert w_spec[0] == 0.0
    assert w_spec[1] == pytest.approx(2.0 * np.exp(-1.0) / (np.sqrt(np.pi) * F_DOM_HZ), rel=1e-12)
    assert w_spec[2] == pytest.approx(8.0 * np.exp(-4.0) / (np.sqrt(np.pi) * F_DOM_HZ), rel=1e-12)


def test_spectrum_peaks_at_dominant_frequency():
    f_hz = np.linspace(0.0, 5.0 * F_DOM_HZ, 100_001, dtype=np.float64)
    w_spec = np.asarray(ricker_spectrum(f_hz, F_DOM_HZ))
    assert f_hz[np.argmax(w_spec)] == pytest.approx(F_DOM_HZ, rel=1e-5)


def test_spectrum_integrates_to_unit_peak():
    # integral of W df over the full line equals the wavelet value at t=0 (1)
    f_hz = np.linspace(0.0, 10.0 * F_DOM_HZ, 2_000_001, dtype=np.float64)
    w_spec = np.asarray(ricker_spectrum(f_hz, F_DOM_HZ))
    assert 2.0 * np.trapezoid(w_spec, f_hz) == pytest.approx(1.0, abs=1e-9)


def test_spectrum_matches_fft_of_sampled_wavelet():
    # rfft(w) == W(f)/dt for a wavelet centered at t=0 (zero phase)
    dt_s = 1.0e-3
    n = 65_536
    t_s = (np.arange(n, dtype=np.float64) - n // 2) * dt_s
    w = ricker_wavelet(t_s, F_DOM_HZ)
    f_hz = np.fft.rfftfreq(n, d=dt_s)
    w_fft = np.fft.rfft(np.fft.ifftshift(w)) * dt_s
    w_spec = ricker_spectrum(f_hz, F_DOM_HZ)
    np.testing.assert_allclose(w_fft.real, w_spec, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(w_fft.imag, 0.0, atol=1e-14)


@pytest.mark.parametrize("bad_f_dom", [0.0, -5.0, np.nan, np.inf])
def test_invalid_dominant_frequency_raises(bad_f_dom):
    with pytest.raises(ValueError, match="f_dom_hz"):
        ricker_wavelet(0.0, bad_f_dom)
    with pytest.raises(ValueError, match="f_dom_hz"):
        ricker_spectrum(0.0, bad_f_dom)
