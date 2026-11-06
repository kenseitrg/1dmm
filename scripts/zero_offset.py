"""M2 demo: zero-offset synthetics for the three wavefield variants (T2, T3).

For each canonical model (``water_bottom`` = T2, ``thick_5layer`` = T3) the
normal-incidence impulse response and the Ricker synthetic are computed for
all three variants (``full`` / ``no_surface`` / ``primaries``) and written to
``output/``:

- ``zero_offset_{model}_{variant}.png`` -- single-panel figure: impulse
  stems + variable-area synthetic for one variant;
- ``zero_offset_{model}_stack.png`` -- the three variants stacked for visual
  comparison (surface-multiple train and interbed events switching off).

Numerics per PLAN.md defaults: dt = 2 ms, Ricker f_dom = 20 Hz, display
records T = 3 s (T2) / 4 s (T3).  The FFT record is extended to 8x the
display window (``n_fft``) so late multiples wrap beyond, not into, the
displayed window.  Numeric checks printed to stdout: water-bottom multiple
train vs the analytic series ``R1 * (-R1)^n`` (alternating polarity, the
free-surface ``R_s = -1`` signature) and variant-difference localization.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mm1d.io import load_blocked_model
from mm1d.reflectivity import ALL_VARIANTS, impulse_response, zero_offset_trace
from mm1d.viz import VARIANT_LABELS, plot_trace_panel, zero_offset_stack_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

DT_S = 2.0e-3
"""Sample interval, s (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Ricker dominant frequency, Hz (PLAN.md numerical defaults)."""

FFT_OVERSAMPLE = 8
"""FFT record extension factor: late multiples wrap beyond the display window."""

MODEL_RECORDS_S: dict[str, float] = {
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


def generate_figures() -> list[Path]:
    """Compute all variant traces for T2/T3 and save the M2 figure set."""
    saved: list[Path] = []
    for model_name, record_s in MODEL_RECORDS_S.items():
        model = load_blocked_model(DATA_DIR / f"{model_name}.csv")
        n_samples = round(record_s / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        t_s = np.arange(n_samples, dtype=np.float64) * DT_S
        panels = {}
        for variant in ALL_VARIANTS:
            impulse = impulse_response(model, variant, DT_S, n_samples, n_fft=n_fft)
            synthetic = zero_offset_trace(model, variant, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft)
            panels[variant] = (impulse, synthetic)

            fig, ax = plt.subplots(figsize=(11.0, 3.6))
            plot_trace_panel(
                ax,
                t_s,
                impulse,
                synthetic,
                title=f"{model.name} -- {VARIANT_LABELS[variant]}",
            )
            ax.set_xlabel("two-way time s")
            fig.tight_layout()
            path = OUTPUT_DIR / f"zero_offset_{model.name}_{variant}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        stack = zero_offset_stack_figure(
            t_s,
            panels,
            suptitle=f"{model.name}: zero-offset variants, Ricker {F_DOM_HZ:g} Hz",
        )
        path = OUTPUT_DIR / f"zero_offset_{model.name}_stack.png"
        stack.savefig(path, dpi=150)
        plt.close(stack)
        saved.append(path)
    return saved


def print_numeric_checks() -> None:
    """Print analytic-anchor checks for T2 (impulse domain, long FFT record)."""
    t2 = load_blocked_model(DATA_DIR / "water_bottom.csv")
    n_samples = round(3.0 / DT_S)
    n_fft = 12_000  # T_fft = 24 s: wraparound below 1e-9
    r1, r2 = t2.reflections_normal
    t_down, t_up = t2.transmissions_normal
    traces = {v: impulse_response(t2, v, DT_S, n_samples, n_fft=n_fft) for v in ALL_VARIANTS}

    print("== water_bottom (T2) analytic anchors ==")
    print(f"R1 = {r1:+.6f}, R2 = {r2:+.6f}, T0_down*T0_up = {t_down[0] * t_up[0]:.6f}")

    def idx(time_s: float) -> int:
        return round(time_s / DT_S)

    print("water-bottom train (full):")
    for n in range(5):
        t = 0.4 * (n + 1)
        got = traces["full"][idx(t)]
        expected = r1 * (-r1) ** n
        print(f"  t = {t:.1f} s: {got:+.6f}  vs  R1*(-R1)^{n} = {expected:+.6f}")
    got = traces["full"][idx(0.9)]
    expected = r2 * t_down[0] * t_up[0]
    print(f"  t = 0.9 s: {got:+.6f}  vs  R2*(1-R1^2)  = {expected:+.6f}")
    pre = slice(0, idx(0.35))
    print("variant differences before the first return (t < 0.35 s):")
    for earlier, later in (("no_surface", "full"), ("primaries", "no_surface")):
        residual = float(np.abs((traces[later] - traces[earlier])[pre]).max())
        print(f"  max |{later} - {earlier}| = {residual:.2e}")
    interbed = (traces["no_surface"] - traces["primaries"])[idx(1.4)]
    print(
        f"interbed (no_surface - primaries) at 1.4 s: {interbed:+.6f}"
        f"  vs  -(1-R1^2)*R1*R2^2 = {-(1 - r1**2) * r1 * r2**2:+.6f}"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the M2 demo: compute synthetics, print checks, save and show figures."""
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
