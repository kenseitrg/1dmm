"""Visual demo of M1 model building (io.py, model.py, units.py).

Generates three figures under ``output/`` (and shows them unless ``--no-show``):

- ``model_demo_models.png`` -- canonical models T1-T3: layer properties vs
  depth and normal-incidence reflection spikes at interface two-way times
  (a preview of the zero-offset primary display, before wavelet convolution).
- ``model_demo_rtheta.png`` -- angle-dependent reflection ``|R(theta)|``
  (solid) and energy-normalized transmission ``T_hat(theta)`` (dashed) per
  interface over 0-50 deg, with critical angles marked as dotted lines; the
  two curves are complementary (``|R|^2 + T_hat^2 = 1``).
- ``model_demo_blocking.png`` -- a synthetic sampled log, written as a
  vp/rho file pair with slightly offset depth grids (merged on depth by
  ``load_log_pair``) and blocked onto a uniform-TWT grid by
  ``block_log_uniform_twt``.

Numeric sanity checks (interface coefficients, worst energy-identity error,
blocking conservation) are printed to stdout.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mm1d.io import block_log_uniform_twt, load_blocked_model, load_log_pair
from mm1d.model import LayeredModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
CANONICAL_NAMES = ("single_interface", "water_bottom", "thick_5layer")

ENERGY_ANGLES_DEG = np.arange(0.0, 50.0 + 1e-9, 0.5)
"""Angle grid for the coefficient figure: 0-50 deg, past T2's critical angle."""


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open interactive figure windows (figures are still saved)",
    )
    return parser


def load_canonical_models() -> list[LayeredModel]:
    """Load the canonical models T1-T3 from data/."""
    return [load_blocked_model(DATA_DIR / f"{name}.csv") for name in CANONICAL_NAMES]


def staircase_xy(
    values: np.ndarray, tops: np.ndarray, bottoms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Staircase polyline for a value-vs-depth log display (value on x, depth on y).

    ``values[i]`` is constant over the depth interval ``[tops[i], bottoms[i]]``;
    consecutive layers are connected at their shared boundary depth.  This is
    the correct orientation for well-log panels (matplotlib's ``step`` only
    steps y as a function of x, i.e. the transposed orientation).
    """
    x = np.repeat(values, 2)
    y = np.column_stack((tops, bottoms)).reshape(-1)
    return x, y


def plot_model_overview(models: list[LayeredModel]) -> Path:
    """Figure 1: layer properties vs depth + normal-incidence R spikes vs TWT."""
    fig, axes = plt.subplots(2, len(models), figsize=(12.5, 7.5))
    for col, model in enumerate(models):
        ax_prop, ax_spike = axes[0, col], axes[1, col]
        tops = np.concatenate(([0.0], model.interface_depth_m))
        # visual half-space extent below the deepest interface
        bottoms = np.append(model.interface_depth_m, model.total_depth_m * 1.12)

        x_vp, depth_y = staircase_xy(model.vp_m_s, tops, bottoms)
        ax_prop.plot(x_vp, depth_y, color="C0", lw=2, label="vp")
        ax_rho = ax_prop.twiny()
        x_rho, _ = staircase_xy(model.rho_kg_m3, tops, bottoms)
        ax_rho.plot(x_rho, depth_y, color="C1", lw=2, ls="-.", label="rho")
        ax_prop.set_title(f"{model.name} ({model.n_layers} layers)")
        ax_prop.set_ylabel("depth m")
        ax_prop.set_xlabel("vp m/s", color="C0")
        ax_rho.set_xlabel("rho kg/m3", color="C1")
        ax_prop.tick_params(axis="x", labelcolor="C0")
        ax_rho.tick_params(axis="x", labelcolor="C1")
        ax_prop.invert_yaxis()
        ax_prop.grid(alpha=0.3)

        for i, (twt, refl) in enumerate(
            zip(model.interface_twt_s, model.reflections_normal, strict=True)
        ):
            color = f"C{i}"
            ax_spike.plot([0.0, refl], [twt, twt], color=color, lw=2)
            ax_spike.plot(refl, twt, "o", color=color)
            # keep the annotation clear of the stem for negative spikes
            if refl >= 0.0:
                offset, align = (5, -3), "left"
            else:
                offset, align = (-5, -3), "right"
            ax_spike.annotate(
                f"{refl:+.3f}",
                (refl, twt),
                xytext=offset,
                textcoords="offset points",
                color=color,
                ha=align,
            )
        ax_spike.axvline(0.0, color="0.5", lw=0.8)
        ax_spike.set_xlim(-0.7, 0.7)
        ax_spike.set_ylim(1.06 * model.interface_twt_s[-1], -0.03 * model.interface_twt_s[-1])
        ax_spike.set_ylabel("TWT s")
        ax_spike.set_xlabel("R (normal incidence)")
        ax_spike.grid(alpha=0.3)
    fig.suptitle("M1 canonical models: layers, impedance contrasts, primary spikes")
    fig.tight_layout()
    path = OUTPUT_DIR / "model_demo_models.png"
    fig.savefig(path, dpi=150)
    return path


def plot_rtheta(models: list[LayeredModel]) -> Path:
    """Figure 2: |R(theta)| and T_hat(theta) per interface with critical angles."""
    angles = ENERGY_ANGLES_DEG
    fig, axes = plt.subplots(1, len(models), figsize=(12.5, 4.2), sharey=True)
    for ax, model in zip(axes, models, strict=True):
        for i in range(model.n_interfaces):
            color = f"C{i}"
            r = np.abs(model.reflection(i, angles))
            t_hat = model.energy_transmission(i, angles)[0]
            label = (
                f"ifc {i}: R$_0$={model.reflections_normal[i]:+.2f} "
                f"@ {model.interface_twt_s[i]:.2f}s"
            )
            ax.plot(angles, r, color=color, label=label)
            ax.plot(angles, t_hat, color=color, ls="--")
            critical = model.critical_angle_deg(i)
            if critical is not None and critical <= angles[-1]:
                ax.axvline(critical, color=color, ls=":", lw=1)
        ax.set_title(model.name)
        ax.set_xlim(0.0, angles[-1])
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("incidence angle deg (in layer above)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|R| solid,  $\\hat{T}$ dashed")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Angle-dependent coefficients; |R|$^2$ + $\\hat{T}^2$ = 1; dotted = critical angle"
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "model_demo_rtheta.png"
    fig.savefig(path, dpi=150)
    return path


def make_demo_log_files() -> tuple[Path, Path]:
    """Write a synthetic vp/rho log pair with offset depth grids to output/."""
    depth = np.arange(0.0, 1500.0 + 0.25, 0.25)
    vp = 1600.0 + 1500.0 * depth / 1500.0 + 150.0 * np.sin(2.0 * np.pi * depth / 260.0)
    rho = 1950.0 + 400.0 * depth / 1500.0 + 80.0 * np.sin(2.0 * np.pi * depth / 310.0 + 1.0)
    vp_path = OUTPUT_DIR / "demo_log_vp.csv"
    rho_path = OUTPUT_DIR / "demo_log_rho.csv"
    np.savetxt(
        vp_path,
        np.column_stack([depth, vp]),
        delimiter=",",
        fmt="%.3f",
        header="depth_m,vp_m_s",
        comments="",
    )
    # deliberately offset depth grid: exercises tolerance-based merging
    np.savetxt(
        rho_path,
        np.column_stack([depth + 0.1, rho]),
        delimiter=",",
        fmt="%.3f",
        header="depth_m,rho_kg_m3",
        comments="",
    )
    return vp_path, rho_path


def plot_blocking_demo() -> tuple[Path, LayeredModel]:
    """Figure 3: pair-merge + uniform-TWT blocking of the synthetic demo log."""
    vp_path, rho_path = make_demo_log_files()
    log = load_log_pair(vp_path, rho_path, depth_tolerance_m=0.5)
    model = block_log_uniform_twt(log, layer_twt_s=0.06, name="demo_blocked")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))
    for ax, log_values, blocked_values, label in (
        (axes[0], log.vp_m_s, model.vp_m_s, "vp m/s"),
        (axes[1], log.rho_kg_m3, model.rho_kg_m3, "rho kg/m3"),
    ):
        ax.plot(log_values, log.depth_m, color="C0", lw=0.6, alpha=0.7, label="sampled log")
        tops = np.concatenate(([0.0], model.interface_depth_m))
        bottoms = np.append(model.interface_depth_m, model.total_depth_m * 1.03)
        x_steps, depth_y = staircase_xy(blocked_values, tops, bottoms)
        ax.plot(
            x_steps,
            depth_y,
            color="C3",
            lw=2,
            label=f"blocked ({len(model.h_m)} layers, 60 ms TWT)",
        )
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_ylabel("depth m")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    layer_tops = np.concatenate(([0.0], model.interface_depth_m))
    layer_bottoms = np.append(model.interface_depth_m, model.total_depth_m)
    for i, twt in enumerate(model.layer_twt_s):
        axes[2].plot([layer_tops[i], layer_bottoms[i]], [twt, twt], color=f"C{i % 10}", lw=2)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("blocked layer TWT s per layer (uniform except last)")
    axes[2].set_ylabel("depth m")
    axes[2].grid(alpha=0.3)
    fig.suptitle("load_log_pair merge + block_log_uniform_twt (half-space duplicates base cell)")
    fig.tight_layout()
    path = OUTPUT_DIR / "model_demo_blocking.png"
    fig.savefig(path, dpi=150)
    return path, model


def print_numeric_checks(models: list[LayeredModel], blocked: LayeredModel) -> None:
    """Print analytic sanity checks for the models and the blocked demo log."""
    angles = np.arange(0.0, 40.0 + 1e-9, 1.0)
    print("== canonical models ==")
    for model in models:
        worst = 0.0
        for i in range(model.n_interfaces):
            r = model.reflection(i, angles)
            t_hat_down, _ = model.energy_transmission(i, angles)
            worst = max(worst, float(np.abs(np.abs(r) ** 2 + np.abs(t_hat_down) ** 2 - 1.0).max()))
        print(model.summary())
        print(f"  worst |R|^2 + T_hat^2 - 1 over 0-40 deg: {worst:.2e}\n")
    print("== blocked demo log ==")
    print(blocked.summary())
    print(f"  base interface R (half-space duplicates base cell): {blocked.reflections_normal[-1]}")


def main(argv: list[str] | None = None) -> None:
    """Run the demo: load models, print checks, save and show figures."""
    args = build_arg_parser().parse_args(argv)
    if args.no_show:
        matplotlib.use("Agg")
    OUTPUT_DIR.mkdir(exist_ok=True)
    models = load_canonical_models()
    paths = [
        plot_model_overview(models),
        plot_rtheta(models),
    ]
    blocking_path, blocked = plot_blocking_demo()
    paths.append(blocking_path)
    print_numeric_checks(models, blocked)
    for path in paths:
        print(f"saved: {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
