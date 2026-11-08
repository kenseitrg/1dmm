"""M4 demo: offset gathers for the three wavefield variants (T1-T3).

For each canonical model (``single_interface`` = T1, ``water_bottom`` = T2,
``thick_5layer`` = T3) the angle-domain plane-wave gathers (theta = 0..40 deg,
1 deg step) are transformed to offset gathers on a uniform offset grid
sized by the deepest primary within the unmuted aperture.  The transform is
the event-agnostic rho-filtered inverse slant stack
(:func:`mm1d.gathers.offset_gather`): every event of the omega-p engine --
primaries, surface and interbed multiples, peg-legs -- lands on its own
Snell-ray moveout curve ``(x(p*), T(p*))`` with no event catalog involved.
Writes to ``output/``:

- ``offset_gather_{model}_{variant}.png`` -- synthetic offset gather as a
  variable-density panel with theoretical moveout overlays: primaries
  (dashed) plus, per model, the canonical multiple curves (T2 water-bottom
  multiples of order 1-2 and a peg-leg; T3 a surface multiple and the soft
  interbed reverberation);
- ``offset_gather_{model}_stack.png`` -- the three variants stacked.

Amplitude convention (stated on every panel): 2-D line-source plane-wave
synthesis -- the |omega| rho-filter reshapes the Ricker wavelet by
sqrt(omega) and event peaks scale as A(p*)/sqrt(2 pi |dx/dp|); no 3-D
geometric spreading; amplitudes are continuous through x = 0.

Artifact control (the synthesis integral is not the naive one): the
slowness aperture is refined internally to the anti-aliasing limit for the
requested offsets and carries a Tukey taper over its last 15 % (below the
post-critical mute where one applies).  Literal FFT wrap-around is
negligible at the 8x record extension (< 3e-5 of peak); the taper
suppresses the acausal radiation of the hard aperture truncation (worst
just below the mute, where |U| grows toward resonance) from ~3 % to ~1 %
of peak, and stationary rays inside the taper zone (the last few degrees,
far offsets of the shallow events) are attenuated gracefully.  The residual
near-offset pre-event ramp (a few % of peak) is the events' own
non-stationary aperture content -- the finite-aperture price of
amplitude-preserving synthesis.

Numeric checks printed to stdout: T1 primary picks on the closed-form
hyperbola and stationary-phase amplitude ratios; the zero-offset trace of
each gather vs the M2 zero-offset trace (event times); T2 water-bottom
multiple picks on its own (steeper) hyperbola vs the same-T0 primary
hyperbola -- V_stk pinned at the water velocity, the corrected
free-surface-multiple expectation (PLAN.md changelog 2026-11-09); and the
angle->offset->angle round-trip correlation/scale.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from mm1d.gathers import (
    SURFACE,
    angle_gather,
    critical_mute_slowness,
    default_offset_grid,
    offset_gather,
    path_offset_time,
    primary_offset_times,
    slant_stack,
    slowness_from_angle,
)
from mm1d.io import load_blocked_model
from mm1d.reflectivity import ALL_VARIANTS
from mm1d.viz import VARIANT_LABELS, offset_gather_stack_figure, plot_offset_gather

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

DT_S = 2.0e-3
"""Sample interval, s (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Ricker dominant frequency, Hz (PLAN.md numerical defaults)."""

FFT_OVERSAMPLE = 8
"""FFT record extension factor: late events delayed by up to p_max x_max must fit."""

ANGLES_DEG = np.arange(0.0, 40.0 + 1.0e-9, 1.0)
"""Incidence-angle (synthesis aperture) grid, deg (PLAN.md defaults)."""

N_OFFSETS = 61
"""Number of offsets on the uniform output grid."""

MODEL_RECORDS_S: dict[str, float] = {
    "single_interface": 1.5,
    "water_bottom": 3.0,
    "thick_5layer": 4.0,
}
"""Display record length per canonical model, s (PLAN.md defaults)."""

AMPLITUDE_NOTE = (
    "2-D line-source plane-wave synthesis: sqrt(omega) wavelet, "
    "A/sqrt(2 pi |dx/dp|) focusing, no 3-D spreading, tapered aperture"
)
"""Amplitude-convention statement carried on every offset-gather panel."""


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open interactive figure windows (figures are still saved)",
    )
    return parser


def model_moveout_curves(model) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Theoretical moveout overlays ``(x(p), T(p))`` for one model.

    Primaries for every interface, plus the canonical multiple family of the
    model: water-bottom multiples (order 1-2) and a peg-leg for T2; a
    first-order surface multiple and the soft-interbed reverberation for T3.
    """
    x_primary, t_primary = primary_offset_times(model, ANGLES_DEG)
    curves = {f"P{k + 1}": (x_primary[:, k], t_primary[:, k]) for k in range(model.n_interfaces)}
    if model.name == "water_bottom":
        curves["M_wb1"] = path_offset_time(model, [SURFACE, 0, SURFACE, 0, SURFACE], ANGLES_DEG)
        curves["M_wb2"] = path_offset_time(model, [SURFACE, 0] * 3 + [SURFACE], ANGLES_DEG)
        curves["peg12"] = path_offset_time(model, [SURFACE, 1, SURFACE, 0, SURFACE], ANGLES_DEG)
    if model.name == "thick_5layer":
        curves["M_s1"] = path_offset_time(model, [SURFACE, 0, SURFACE, 0, SURFACE], ANGLES_DEG)
        curves["interbed23"] = path_offset_time(model, [SURFACE, 2, 1, 2, SURFACE], ANGLES_DEG)
    return curves


def generate_figures() -> list[Path]:
    """Compute all variant offset gathers for T1/T2/T3 and save the M4 figure set."""
    saved: list[Path] = []
    for model_name, record_s in MODEL_RECORDS_S.items():
        model = load_blocked_model(DATA_DIR / f"{model_name}.csv")
        n_samples = round(record_s / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        t_s = np.arange(n_samples, dtype=np.float64) * DT_S
        offsets = default_offset_grid(model, ANGLES_DEG, n_offsets=N_OFFSETS)
        moveout = model_moveout_curves(model)
        panels = {}
        for variant in ALL_VARIANTS:
            panels[variant] = offset_gather(
                model, variant, ANGLES_DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
            )

            fig, ax = plt.subplots(figsize=(11.0, 5.2))
            plot_offset_gather(
                ax,
                offsets,
                t_s,
                panels[variant],
                title=f"{model.name} -- {VARIANT_LABELS[variant]}  ({AMPLITUDE_NOTE})",
                moveout=moveout,
            )
            ax.set_xlabel("offset m")
            fig.tight_layout()
            path = OUTPUT_DIR / f"offset_gather_{model.name}_{variant}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        stack = offset_gather_stack_figure(
            offsets,
            t_s,
            panels,
            suptitle=f"{model.name}: offset gathers, Ricker {F_DOM_HZ:g} Hz -- {AMPLITUDE_NOTE}",
            moveout=moveout,
        )
        path = OUTPUT_DIR / f"offset_gather_{model.name}_stack.png"
        stack.savefig(path, dpi=150)
        plt.close(stack)
        saved.append(path)
    return saved


def print_numeric_checks() -> None:
    """Print the M4 acceptance anchors: T1 hyperbola, zero-offset trace, T2 multiple."""
    t1 = load_blocked_model(DATA_DIR / "single_interface.csv")
    t2 = load_blocked_model(DATA_DIR / "water_bottom.csv")

    print("== T1 primary: picks on the closed-form hyperbola + stationary-phase amplitudes ==")
    h, v = 500.0, 2000.0
    theta = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    offsets = 2.0 * h * np.tan(np.radians(theta))
    times = 2.0 * h / (v * np.cos(np.radians(theta)))
    n_samples = round(1.5 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    gather = offset_gather(
        t1, "primaries", ANGLES_DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
    )
    freq = np.fft.rfftfreq(n_fft, d=DT_S)
    omega = 2.0 * np.pi * freq
    from mm1d.wavelet import ricker_spectrum

    wavelet_ref = np.fft.irfft(
        np.sqrt(omega) * np.asarray(ricker_spectrum(freq, F_DOM_HZ)) / DT_S, n=n_fft
    )
    p0 = float(np.max(np.abs(hilbert(wavelet_ref))))
    p_grid = np.atleast_1d(np.asarray(slowness_from_angle(theta, v)))
    q = np.sqrt(1.0 / v**2 - p_grid**2)
    dx_dp = 2.0 * h / (v**2 * q**3)
    print("theta   x m      T theory   T pick    amp pred   amp got   ratio  (>25 deg edge)")
    for theta_i, x_i, t_i, trace, xprime in zip(theta, offsets, times, gather, dx_dp, strict=True):
        env = np.abs(hilbert(trace))
        i0 = round(float(t_i) / DT_S)
        amp = float(env[i0 - 30 : i0 + 30].max())
        pick = (i0 - 30 + int(np.argmax(env[i0 - 30 : i0 + 30]))) * DT_S
        r1 = float(np.real(t1.reflection(0, theta_i)))
        pred = r1 / np.sqrt(2.0 * np.pi * xprime) * p0
        print(
            f"{theta_i:4.0f} {x_i:8.1f} {t_i:10.4f} {pick:9.4f} "
            f"{pred:10.6f} {amp:9.6f} {amp / pred:8.4f}"
        )

    print("\n== Zero-offset trace (x = 0) vs M2: event-time picks ==")
    for model_name, record_s, events in (
        ("water_bottom", 3.0, (0.4, 0.8, 0.9, 1.2)),
        ("thick_5layer", 4.0, (0.5, 1.0, 1.3, 1.8)),
    ):
        model = load_blocked_model(DATA_DIR / f"{model_name}.csv")
        n_samples = round(record_s / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        trace = offset_gather(
            model, "full", ANGLES_DEG, np.array([0.0]), DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
        )[0]
        env = np.abs(hilbert(trace))
        picks = []
        for t0 in events:
            i0 = round(t0 / DT_S)
            picks.append((i0 - 30 + int(np.argmax(env[i0 - 30 : i0 + 30]))) * DT_S)
        print(f"  {model.name:16s} full: theory {events} -> picks {np.round(picks, 4)}")

    print("\n== T2 water-bottom multiple: its hyperbola (V = v_w) vs same-T0 primary hyperbola ==")
    n_samples = round(3.0 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    theta = np.array([5.0, 15.0, 25.0])
    x_mult, t_mult = path_offset_time(t2, [SURFACE, 0, SURFACE, 0, SURFACE], theta)
    gather = offset_gather(t2, "full", ANGLES_DEG, x_mult, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft)
    v_rms = float(np.sqrt((0.4 * 1500.0**2 + 0.4 * 2400.0**2) / 0.8))
    print(f"  V_stk(multiple) = 1500 m/s vs Dix V_rms(0.8 s) = {v_rms:.0f} m/s (steeper)")
    print("  x m    T mult   T pick   T same-T0 primary   separation ms")
    for x_i, t_i, trace in zip(x_mult, t_mult, gather, strict=True):
        env = np.abs(hilbert(trace))
        i0 = round(float(t_i) / DT_S)
        pick = (i0 - 30 + int(np.argmax(env[i0 - 30 : i0 + 30]))) * DT_S
        t_primary = np.sqrt(0.8**2 + (x_i / v_rms) ** 2)
        print(
            f"  {x_i:6.1f} {t_i:7.4f} {pick:8.4f} {t_primary:19.4f} "
            f"{1e3 * (t_primary - t_i):+10.1f}"
        )

    print("\n== Round trip: angle gather -> offset gather -> forward slant stack ==")
    for model, variant, angle_max in ((t1, "primaries", 30.0), (t2, "full", 27.0)):
        n_samples = round((1.5 if model is t1 else 3.0) / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        original = angle_gather(model, variant, ANGLES_DEG, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft)
        offsets = default_offset_grid(model, ANGLES_DEG, n_offsets=N_OFFSETS)
        synthesized = offset_gather(
            model, variant, ANGLES_DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
        )
        p_grid = np.atleast_1d(np.asarray(slowness_from_angle(ANGLES_DEG, model.vp_m_s[0])))
        recovered = slant_stack(synthesized, offsets, p_grid, DT_S, n_fft=n_fft)
        angle_mask = angle_max >= ANGLES_DEG
        time_mask = np.arange(n_samples) * DT_S <= 0.7 * n_samples * DT_S
        got = recovered[np.ix_(angle_mask, time_mask)]
        want = original[np.ix_(angle_mask, time_mask)]
        corr = float(np.corrcoef(got.ravel(), want.ravel())[0, 1])
        scale = float(np.sum(got * want) / np.sum(want * want))
        print(
            f"  {model.name:16s} {variant:9s}: corr = {corr:.4f}, scale = {scale:.4f} "
            f"(finite-aperture corrections; event times exact)"
        )

    p_mute = critical_mute_slowness(load_blocked_model(DATA_DIR / "water_bottom.csv"))
    print(f"\n(post-critical mute: p = {p_mute:.3e} s/m -> theta = 30 deg in water for T2/T3)")


def main(argv: list[str] | None = None) -> None:
    """Run the M4 demo: compute gathers, print checks, save and show figures."""
    args = build_arg_parser().parse_args(argv)
    if args.no_show:
        matplotlib.use("Agg")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print_numeric_checks()
    for path in generate_figures():
        print(f"saved: {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
