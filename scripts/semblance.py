"""M5 demo: Neidell-Taner semblance panels for the three wavefield variants (T1-T3).

For each canonical model (``single_interface`` = T1, ``water_bottom`` = T2,
``thick_5layer`` = T3) the offset gathers (M4 synthesis, Ricker band) are
scanned over ``(T0, V_stk)`` with sliding-window semblance on hyperbolic
moveout (:func:`mm1d.semblance.semblance_panel`), one panel per variant on a
shared grid.  Writes to ``output/``:

- ``semblance_{model}_{variant}.png`` -- the panel with reference overlays:
  the Dix RMS curve of the primaries (white dashed) and, for T2, the water
  velocity pin line of the surface-multiple family;
- ``semblance_{model}_stack.png`` -- the three variants stacked with one
  colorbar.

Scan design (stated on every panel):

- **Aperture**: offsets 0..``SPREAD_CAP_M_PER_S`` x T0(deepest primary)
(= 1.1 km/s x T0 ~ 1.1 m per ms of two-way time) -- a moderate spread,
because Dix velocities are small-spread stacking velocities: at the full M4
aperture the non-hyperbolic Snell moveout drags the maxima ~25-35 m/s above
Dix; at this aperture the hyperbola fits to a few ms and picks track Dix
within about one velocity cell.
- **Slowness bound**: trial hyperbolas are scored only on offsets whose local
  slowness ``p = dt/dx`` is inside the taper-clean synthesis aperture
  (:func:`mm1d.semblance.scan_slowness_bound` = ``(1 - 0.15) * p_end``) --
  beyond the taper the M4 synthesis documents attenuated, phase-shifted
  event tails (e.g. P1 of T2 is clean only to ~280 m).
- **Noise floor**: zero by default. --noise-fraction .01 enables an explicitly
  labelled seeded experiment, not artifact removal. Noiseless coherence ties
  may broaden peaks; unrestricted 2-D argmax is not a unique velocity pick.
- **Backend**: --backend ray uses untapered ray support and sample validity,
  NOT the legacy taper-clean restriction. Outputs include fold maps and NPZ
  arrays, plus compact event tables separate from the full synthesis catalog.
  Ray filenames carry _ray; the default backend remains slant.
- **Gate**: one dominant period (50 ms at 20 Hz), the Neidell-Taner choice;
  the sqrt(omega)-reshaped wavelet keeps isolated events coherent (S -> 1)
  while 100 ms-separated neighbours do not interfere.
- **Resolution floor**: at the taper-clean aperture the shallowest interface
  of each model (``x_max / T0 ~ 0.7``) resolves its stacking velocity only
  to ~+/-100 m/s at 20 Hz; deep interfaces pin to Dix within about one
  velocity cell.  The shallow-interface picks therefore carry the resolution
  tolerance in the printed checks.

Numeric checks printed to stdout: T2 ``primaries`` maxima vs Dix at both
interfaces (the M5 acceptance anchor; shallow-interface resolution floor
noted); the T2 ``full`` water-bottom multiple train (orders 1-4) pinned at
the water velocity, *below* the same-T0 Dix primary trend (the
free-surface-multiple / radon-separability signature); the peg-leg ridge
between the two trends; T1 primary + surface multiple both pinned at the
layer velocity; T3 primaries vs Dix.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mm1d.events import display_events, enumerate_events
from mm1d.gathers import SURFACE, offset_gather, path_offset_time
from mm1d.io import load_blocked_model
from mm1d.ray import ray_offset_gather
from mm1d.reflectivity import ALL_VARIANTS
from mm1d.semblance import (
    default_t0_grid,
    default_velocity_grid,
    dix_rms_velocities,
    scan_slowness_bound,
    semblance_with_fold,
)
from mm1d.viz import (
    VARIANT_LABELS,
    SemblanceOverlay,
    plot_semblance,
    semblance_stack_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

DT_S = 2.0e-3
"""Sample interval, s (PLAN.md numerical defaults)."""

F_DOM_HZ = 20.0
"""Ricker dominant frequency, Hz (PLAN.md numerical defaults)."""

FFT_OVERSAMPLE = 8
"""FFT record extension factor (M2-M4 practice: late multiples wrap outside)."""

ANGLES_DEG = np.arange(0.0, 40.0 + 1.0e-9, 1.0)
"""Incidence-angle (synthesis aperture) grid, deg (PLAN.md defaults)."""

N_OFFSETS = 61
"""Number of offsets on the uniform scan gather grid."""

GATE_S = 0.05
"""Semblance gate: one dominant period of the Ricker source, s."""

SPREAD_CAP_M_PER_S = 1100.0
"""Scan aperture velocity: max offset = SPREAD_CAP_M_PER_S * T0(deepest primary)
(m/s x s -> m; ~1.1 m per ms of two-way time -- a moderate spread, see the
module docstring)."""

NOISE_FLOOR_FRACTION = 0.0
"""Seeded incoherent noise floor added to scan gathers, fraction of peak."""

NOISE_SEED = 20261110
"""Seed of the noise floor (deterministic figures and checks)."""

MODEL_RECORDS_S: dict[str, float] = {
    "single_interface": 1.5,
    "water_bottom": 3.0,
    "thick_5layer": 4.0,
}
"""Display record length per canonical model, s (PLAN.md defaults)."""

SCAN_NOTE = "Neidell-Taner, 50 ms gate, hyperbolic, noiseless by default"
"""Scan-convention statement carried on every semblance panel (kept short:
 it must fit the single-panel figure width)."""


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open interactive figure windows (figures are still saved)",
    )
    parser.add_argument("--backend", choices=("slant", "ray"), default="slant")
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=NOISE_FLOOR_FRACTION,
        help="optional seeded noise experiment; default raw noiseless gathers",
    )
    return parser


def scan_setup(model, record_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scan offsets (m), T0 grid (s), and velocity grid (m/s) for one model."""
    t0_grid = default_t0_grid(record_s)
    v_grid = default_velocity_grid(model)
    x_max = SPREAD_CAP_M_PER_S * float(model.interface_twt_s[-1])
    offsets = np.linspace(0.0, x_max, N_OFFSETS)
    return offsets, t0_grid, v_grid


def reference_overlays(model, variant="full") -> SemblanceOverlay:
    """Return the reference ``(t0, v)`` overlay curves: Dix primaries + velocity pins."""
    dix_t0 = model.interface_twt_s
    dix_v = dix_rms_velocities(model)
    entries: list[tuple[str, tuple[np.ndarray, ...] | tuple[np.ndarray, np.ndarray, str]]] = [
        ("Dix V_rms", (dix_t0, dix_v, "wo--"))
    ]
    if model.name == "water_bottom" and variant == "full":
        # surface multiples of the water bottom are pinned at the water
        # velocity at every order -- a horizontal reference line (dotted, to
        # distinguish the pin from the Dix curve)
        # points ordered right-to-left so the label lands at the line's right
        # end, clear of the Dix label at the curve start
        entries.append(("v_w pin", (np.array([2.05, 0.35]), np.array([1500.0, 1500.0]), "w:")))
    return dict(entries)


def dix_trend_v(model, t0_s: float | np.ndarray) -> np.ndarray:
    """Dix RMS velocity of a hypothetical primary at T0, m/s (constant-v layers).

    Interval-velocity interpolation of :func:`mm1d.semblance.dix_rms_velocities`:
    within layer ``k`` the remaining two-way time ``t`` adds ``t * v_k^2`` to
    the Dix sums (the same-T0 primary trend the multiples are compared
    against).
    """
    t0 = np.atleast_1d(np.asarray(t0_s, dtype=np.float64))
    twt = model.layer_twt_s
    vp = model.vp_m_s[:-1]
    cum_t = np.cumsum(twt)
    k = np.clip(np.searchsorted(cum_t, t0, side="right"), 0, twt.size - 1)
    t_above = np.where(k > 0, cum_t[np.maximum(k - 1, 0)], 0.0)
    sum_tv2 = (
        np.where(
            k > 0,
            np.concatenate(([0.0], np.cumsum(twt * vp**2)))[k],
            0.0,
        )
        + (t0 - t_above) * vp[k] ** 2
    )
    return np.sqrt(sum_tv2 / t0)


def local_pick(
    panel: np.ndarray,
    t0_grid: np.ndarray,
    v_grid: np.ndarray,
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
    flat = int(np.argmax(window))
    i_v, i_t = np.unravel_index(flat, window.shape)
    v_indices = np.flatnonzero(v_mask)
    t_indices = np.flatnonzero(t_mask)
    return float(t0_grid[t_indices[i_t]]), float(v_grid[v_indices[i_v]]), float(window[i_v, i_t])


def scan_gather(model, variant, offsets, n_samples, n_fft, backend="slant", noise_fraction=0.0):
    """Return gather, sample validity and backend-specific slowness bound (s/m)."""
    if not np.isfinite(noise_fraction) or noise_fraction < 0:
        raise ValueError("noise fraction must be finite and nonnegative")
    if backend == "ray":
        result = ray_offset_gather(model, variant, offsets, DT_S, n_samples, F_DOM_HZ)
        gather, valid, p_bound = result.traces, result.valid_samples, result.p_max_s_m
        # Display selection is intentionally separate from the synthesis set.
        table = ["| path | family | T0 s | RMS m/s |", "|---|---|---:|---:|"]
        table += [
            f"| {e.turning_points} | {e.family} | {e.t0_s:.4f} | {e.vrms_m_s:.1f} |"
            for e in display_events(model, result.catalog)
        ]
        (OUTPUT_DIR / f"events_{model.name}_{variant}.md").write_text(
            f"Bounds: {result.catalog.reached_bounds}; "
            f"synthesis paths: {len(result.catalog.events)}. "
            "Display only: order <=3, |A(0)| >=0.01. "
            "RMS is small-spread, not finite-offset fit.\n\n" + "\n".join(table) + "\n"
        )
    else:
        gather = offset_gather(
            model, variant, ANGLES_DEG, offsets, DT_S, n_samples, F_DOM_HZ, n_fft=n_fft
        )
        valid = np.ones(gather.shape, dtype=bool)
        p_bound = scan_slowness_bound(model, ANGLES_DEG)
    rng = np.random.default_rng(NOISE_SEED)
    sigma = noise_fraction * float(np.abs(gather).max())
    return gather + rng.normal(0.0, sigma, gather.shape), valid, p_bound


def generate_figures(backend="slant", noise_fraction=0.0) -> list[Path]:
    """Compute all variant semblance panels for T1/T2/T3 and save the M5 figure set."""
    saved: list[Path] = []
    for model_name, record_s in MODEL_RECORDS_S.items():
        model = load_blocked_model(DATA_DIR / f"{model_name}.csv")
        n_samples = round(record_s / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        offsets, t0_grid, v_grid = scan_setup(model, record_s)
        panels = {}
        overlays = {}
        suffix = "_ray" if backend == "ray" else ""
        if noise_fraction:
            suffix += f"_noise{noise_fraction:g}"
        note = (
            f"{backend} backend; 50 ms gate; noise {100 * noise_fraction:g}%\n"
            "fold/support saved separately"
        )
        for variant in ALL_VARIANTS:
            gather, valid, p_bound = scan_gather(
                model, variant, offsets, n_samples, n_fft, backend, noise_fraction
            )
            scan = semblance_with_fold(
                gather,
                offsets,
                DT_S,
                t0_grid,
                v_grid,
                GATE_S,
                p_mute_s_m=p_bound,
                valid_samples=valid,
            )
            panels[variant] = scan.panel
            overlay = dict(reference_overlays(model, variant))
            if backend == "ray":
                selected = display_events(model, enumerate_events(model, variant, record_s))
                for family in ("surface", "interbed", "mixed"):
                    family_events = [e for e in selected if e.family == family]
                    if family_events:
                        overlay[f"{family} RMS"] = (
                            np.array([e.t0_s for e in family_events]),
                            np.array([e.vrms_m_s for e in family_events]),
                            "w.",
                        )
            overlays[variant] = overlay
            np.savez(
                OUTPUT_DIR / f"semblance_{model.name}_{variant}{suffix}.npz",
                panel=scan.panel,
                fold=scan.fold,
                t0_s=t0_grid,
                velocity_m_s=v_grid,
            )
            fold_fig, fold_ax = plt.subplots(figsize=(6.8, 6.4), layout="constrained")
            image = fold_ax.pcolormesh(v_grid, t0_grid, scan.fold.T, vmin=0, vmax=N_OFFSETS)
            fold_ax.invert_yaxis()
            fold_ax.set(
                xlabel="stacking velocity m/s",
                ylabel="zero-offset time s",
                title=f"{model.name} / {variant}: contributing traces\n{note}",
            )
            fold_fig.colorbar(image, ax=fold_ax, label="fold")
            fold_path = OUTPUT_DIR / f"semblance_fold_{model.name}_{variant}{suffix}.png"
            fold_fig.savefig(fold_path, dpi=150)
            plt.close(fold_fig)
            saved.append(fold_path)

            fig, ax = plt.subplots(figsize=(6.8, 6.4), layout="constrained")
            plot_semblance(
                ax,
                v_grid,
                t0_grid,
                panels[variant],
                title=f"{model.name} -- {VARIANT_LABELS[variant]}\n{note}",
                overlay=overlay,
            )
            path = OUTPUT_DIR / f"semblance_{model.name}_{variant}{suffix}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        stack = semblance_stack_figure(
            v_grid,
            t0_grid,
            panels,
            suptitle=f"{model.name}: semblance panels\n{note}",
            overlay_by_variant=overlays,
        )
        path = OUTPUT_DIR / f"semblance_{model.name}_stack{suffix}.png"
        stack.savefig(path, dpi=150)
        plt.close(stack)
        saved.append(path)
    return saved


def print_numeric_checks(backend="slant", noise_fraction=0.0) -> None:
    """Print the M5 acceptance anchors: Dix primaries, pinned multiples, peg-leg."""
    t1 = load_blocked_model(DATA_DIR / "single_interface.csv")
    t2 = load_blocked_model(DATA_DIR / "water_bottom.csv")
    t3 = load_blocked_model(DATA_DIR / "thick_5layer.csv")

    def build_panels(model, record_s):
        n_samples = round(record_s / DT_S)
        n_fft = FFT_OVERSAMPLE * n_samples
        offsets, t0_grid, v_grid = scan_setup(model, record_s)
        panels = {}
        for variant in ALL_VARIANTS:
            gather, valid, p_bound = scan_gather(
                model, variant, offsets, n_samples, n_fft, backend, noise_fraction
            )
            panels[variant] = semblance_with_fold(
                gather,
                offsets,
                DT_S,
                t0_grid,
                v_grid,
                GATE_S,
                p_mute_s_m=p_bound,
                valid_samples=valid,
            ).panel
        return offsets, t0_grid, v_grid, panels

    _, t0_t2, v_t2, panels_t2 = build_panels(t2, MODEL_RECORDS_S["water_bottom"])
    dix_t2 = dix_rms_velocities(t2)

    print("== T2 primaries: free local argmax vs Dix (not unique on noiseless data) ==")
    print("  T0 Dix   Dix V    pick T0  pick V   S")
    for t0, v_dix in zip(t2.interface_twt_s, dix_t2, strict=True):
        p_t0, p_v, p_s = local_pick(panels_t2["primaries"], t0_t2, v_t2, float(t0), float(v_dix))
        print(f"  {t0:.2f} {v_dix:7.1f} {p_t0:9.4f} {p_v:8.1f} {p_s:6.3f}")

    print("\n== T2 full: water-bottom multiple train pinned at v_w (below Dix trend) ==")
    print("  T0 = (n+1)*0.4   pinned pick V   Dix V_rms(T0)   delta")
    for order in range(1, 5):
        t0 = (order + 1) * 0.4
        _, p_v, _ = local_pick(panels_t2["full"], t0_t2, v_t2, t0, 1500.0)
        v_dix_t0 = float(dix_trend_v(t2, t0)[0])
        print(f"  {t0:5.1f} s       {p_v:9.1f}       {v_dix_t0:7.1f}   {p_v - 1500.0:+6.1f}")

    print("\n== T2 full: peg-leg peg12 ridge between the pin and the Dix trend ==")
    _, t_peg = path_offset_time(t2, [SURFACE, 1, SURFACE, 0, SURFACE], np.array([0.0]))
    t0_peg = float(t_peg[0])
    # stacking velocity from the path legs at zero offset: 4 water legs,
    # 2 sediment legs (V_stk = sqrt(sum t_leg v_leg^2 / sum t_leg))
    t_water = 4.0 * float(t2.h_m[0] / t2.vp_m_s[0])
    t_sed = 2.0 * float(t2.h_m[1] / t2.vp_m_s[1])
    v_peg = float(
        np.sqrt((t_water * t2.vp_m_s[0] ** 2 + t_sed * t2.vp_m_s[1] ** 2) / (t_water + t_sed))
    )
    p_t0, p_v, p_s = local_pick(panels_t2["full"], t0_t2, v_t2, t0_peg, v_peg)
    print(
        f"  theory ({t0_peg:.2f} s, {v_peg:.1f} m/s) "
        f"-> pick ({p_t0:.3f} s, {p_v:.1f} m/s, S={p_s:.3f})"
    )

    _, t0_t1, v_t1, panels_t1 = build_panels(t1, MODEL_RECORDS_S["single_interface"])
    print("\n== T1: primary (0.5 s) and surface multiple (1.0 s) both at v1 = 2000 ==")
    for t0, label in ((0.5, "P1"), (1.0, "M1")):
        p_t0, p_v, p_s = local_pick(panels_t1["full"], t0_t1, v_t1, t0, 2000.0)
        print(f"  {label}: pick ({p_t0:.3f} s, {p_v:.1f} m/s, S={p_s:.3f})")

    _, t0_t3, v_t3, panels_t3 = build_panels(t3, MODEL_RECORDS_S["thick_5layer"])
    dix_t3 = dix_rms_velocities(t3)
    print("\n== T3 primaries: free local argmax vs Dix (noiseless ties can bias picks) ==")
    print("  T0 Dix   Dix V    pick T0  pick V   S")
    for t0, v_dix in zip(t3.interface_twt_s, dix_t3, strict=True):
        p_t0, p_v, p_s = local_pick(panels_t3["primaries"], t0_t3, v_t3, float(t0), float(v_dix))
        print(f"  {t0:.2f} {v_dix:7.1f} {p_t0:9.4f} {p_v:8.1f} {p_s:6.3f}")

    print(
        f"\nBackend={backend}, noise={100 * noise_fraction:g}%. "
        "RMS anchors are small-spread; noiseless coherence ties are not unique velocity picks."
    )


def main(argv: list[str] | None = None) -> None:
    """Run the M5 demo: compute panels, print checks, save and show figures."""
    args = build_arg_parser().parse_args(argv)
    if args.no_show:
        matplotlib.use("Agg")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print_numeric_checks(args.backend, args.noise_fraction)
    for path in generate_figures(args.backend, args.noise_fraction):
        print(f"saved: {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
