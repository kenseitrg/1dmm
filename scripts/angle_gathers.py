"""M3 demo: angle-domain gathers for the three wavefield variants (T1-T3).

For each canonical model (``single_interface`` = T1, ``water_bottom`` = T2,
``thick_5layer`` = T3) the plane-wave omega-p engine produces impulse and
Ricker-synthetic angle gathers (theta = 0..40 deg, 1 deg step) for all three
variants (``full`` / ``no_surface`` / ``primaries``) and writes to ``output/``:

- ``angle_gather_{model}_{variant}.png`` -- synthetic gather as a
  variable-density panel with the analytic primary-intercept curves
  (dashed) and the post-critical mute boundary (dotted) overlaid;
- ``angle_gather_{model}_stack.png`` -- the three variants stacked for
  visual comparison.

Events in a plane-wave gather appear at their *intercept* times
``tau(p) = 2 sum q_j h_j`` (decreasing with angle).  Amplitudes are
plane-wave pressures: no geometric spreading, no obliquity factor -- stated
on every panel.  Numerics per PLAN.md defaults: dt = 2 ms, Ricker 20 Hz,
FFT record extended 8x so late multiples wrap beyond the display window.
Traces beyond the smallest interface-critical slowness (``p = 1/v_max``,
i.e. theta > 30 deg for T2/T3) are muted: there the stack is a lossless
total reflector and the ``full`` free-surface series resonates.

Numeric checks printed to stdout: T1 first-arrival amplitude vs ``R1(theta)``
and arrival time vs ``0.5 s * cos(theta)`` (envelope picks), the theta = 0
column vs the M2 zero-offset trace, and the analytic T1 multiple-train
amplitude ``R1(theta) * (-R1(theta))^n`` in the frequency domain.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from mm1d.gathers import (
    angle_gather,
    angle_impulse_gather,
    critical_mute_slowness,
    primary_intercept_times,
    slowness_from_angle,
)
from mm1d.io import load_blocked_model
from mm1d.reflectivity import ALL_VARIANTS, impulse_response, stack_response
from mm1d.viz import VARIANT_LABELS, angle_gather_stack_figure, plot_angle_gather

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

DT_S = 2.0e-3
"""Sample interval, s (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Ricker dominant frequency, Hz (PLAN.md numerical defaults)."""

FFT_OVERSAMPLE = 8
"""FFT record extension factor: late multiples wrap beyond the display window."""

ANGLES_DEG = np.arange(0.0, 40.0 + 1.0e-9, 1.0)
"""Incidence-angle grid (PLAN.md numerical defaults: 0-40 deg, 1 deg step)."""

MODEL_RECORDS_S: dict[str, float] = {
    "single_interface": 1.5,
    "water_bottom": 3.0,
    "thick_5layer": 4.0,
}
"""Display record length per canonical model, s (PLAN.md defaults)."""


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open interactive figure windows (figures are still saved)",
    )
    return parser


def mute_angle_deg(model) -> float | None:
    """Mute-boundary angle in the surface layer, degrees (None if no critical)."""
    p_mute = critical_mute_slowness(model)
    if p_mute is None:
        return None
    return float(np.degrees(np.arcsin(p_mute * model.vp_m_s[0])))


def generate_figures() -> list[Path]:
    """Compute all variant angle gathers for T1/T2/T3 and save the M3 figure set."""
    saved: list[Path] = []
    for model_name, record_s in MODEL_RECORDS_S.items():
        model = load_blocked_model(DATA_DIR / f"{model_name}.csv")
        n_samples = round(record_s / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        t_s = np.arange(n_samples, dtype=np.float64) * DT_S
        intercept = primary_intercept_times(model, ANGLES_DEG)
        moveout = {f"P{k + 1}": intercept[:, k] for k in range(model.n_interfaces)}
        mute = mute_angle_deg(model)
        panels = {}
        for variant in ALL_VARIANTS:
            panels[variant] = angle_gather(
                model, variant, ANGLES_DEG, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
            )

            fig, ax = plt.subplots(figsize=(11.0, 5.2))
            plot_angle_gather(
                ax,
                ANGLES_DEG,
                t_s,
                panels[variant],
                title=(
                    f"{model.name} -- {VARIANT_LABELS[variant]}  "
                    f"(plane-wave pressure amplitudes, no geometric spreading)"
                ),
                moveout=moveout,
                mute_angle_deg=mute,
            )
            ax.set_xlabel("incidence angle deg")
            fig.tight_layout()
            path = OUTPUT_DIR / f"angle_gather_{model.name}_{variant}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        stack = angle_gather_stack_figure(
            ANGLES_DEG,
            t_s,
            panels,
            suptitle=(
                f"{model.name}: angle gathers, Ricker {F_DOM_HZ:g} Hz -- "
                f"plane-wave pressure amplitudes, no geometric spreading"
            ),
            moveout=moveout,
            mute_angle_deg=mute,
        )
        path = OUTPUT_DIR / f"angle_gather_{model.name}_stack.png"
        stack.savefig(path, dpi=150)
        plt.close(stack)
        saved.append(path)
    return saved


def print_numeric_checks() -> None:
    """Print analytic-anchor checks for T1 and the theta = 0 column (T2/T3)."""
    print("== T1 (single_interface): first arrival vs R1(theta) and 0.5 s cos(theta) ==")
    t1 = load_blocked_model(DATA_DIR / "single_interface.csv")
    n_samples = round(1.5 / DT_S)
    n_fft = FFT_OVERSAMPLE * n_samples
    angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    gather = angle_gather(t1, "full", angles, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft)
    print("theta   R1(theta)   env. peak   0.5cos(theta)   env. time")
    for row, theta in zip(gather, angles, strict=True):
        envelope = np.abs(hilbert(row))
        i_peak = int(np.argmax(envelope))
        r1 = float(np.real(t1.reflection(0, theta)))
        tau = 0.5 * np.cos(np.radians(theta))
        print(
            f"{theta:4.0f}   {r1:+9.6f}   {envelope[i_peak]:9.6f}   "
            f"{tau:12.6f}   {i_peak * DT_S:.4f}"
        )

    # T1 full-variant multiple train in the frequency domain (exact, any angle)
    freq = np.linspace(0.0, 200.0, 401)
    print("\n== T1 full variant vs analytic series R e^{-2i w t} / (1 + R e^{-2i w t}) ==")
    for theta in (20.0, 40.0):
        r1 = np.conj(complex(t1.reflection(0, theta)))
        p = float(slowness_from_angle(theta, t1.vp_m_s[0]))
        tau = 2.0 * t1.h_m[0] * np.cos(np.radians(theta)) / t1.vp_m_s[0]
        x = r1 * np.exp(-2j * np.pi * freq * tau)
        expected = x / (1.0 + x)
        got = stack_response(t1, "full", freq, p)
        err = np.max(np.abs(got - expected))
        print(f"  theta = {theta:4.0f} deg: max |S - series| = {err:.2e}")

    print("\n== theta = 0 column vs M2 zero-offset impulse trace (max abs diff) ==")
    for model_name, record_s in MODEL_RECORDS_S.items():
        model = load_blocked_model(DATA_DIR / f"{model_name}.csv")
        if model_name == "single_interface":
            continue  # T1 shown above; T2/T3 exercise multi-layer stacks
        n_samples = round(record_s / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        gather = angle_impulse_gather(model, "full", np.array([0.0]), DT_S, n_samples, n_fft=n_fft)
        reference = impulse_response(model, "full", DT_S, n_samples, n_fft=n_fft)
        print(f"  {model.name:16s} full: {np.max(np.abs(gather[0] - reference)):.2e}")


def main(argv: list[str] | None = None) -> None:
    """Run the M3 demo: compute gathers, print checks, save and show figures."""
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
