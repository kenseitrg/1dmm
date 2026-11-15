"""Reproduce numerical budgets and inspect weak/near-offset windows, without noise.

Comparison display scale: each backend normalized by its own full-gather peak,
shared across offsets/zoom windows. The two backends have different wavelets
and amplitude conventions; samplewise equality is deliberately NOT asserted.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from tracemalloc import get_traced_memory, start, stop

import matplotlib.pyplot as plt
import numpy as np

from mm1d.gathers import _aperture_weights, offset_gather
from mm1d.io import load_blocked_model
from mm1d.ray import ray_offset_gather
from mm1d.reflectivity import ALL_VARIANTS, zero_offset_trace

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Write output/numerical_checks.json and diagnostic_windows.png; optionally show."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    angles = np.arange(41.0)
    offsets = np.array([0.0, 200.0, 600.0])
    dt, n = 0.002, 2000
    times = np.arange(n) * dt
    report: dict[str, object] = {}
    for name in ("single_interface", "water_bottom", "thick_5layer"):
        model = load_blocked_model(ROOT / "data" / f"{name}.csv")
        for variant in ALL_VARIANTS:
            start()
            before = perf_counter()
            ray = ray_offset_gather(model, variant, offsets, dt, n, 20.0)
            elapsed = perf_counter() - before
            _, memory = get_traced_memory()
            stop()
            reference = zero_offset_trace(model, variant, dt, n, 20.0, n_fft=32 * n)
            report[f"{name}/{variant}"] = {
                "ray_seconds": elapsed,
                "traced_peak_bytes": memory,
                "paths": len(ray.catalog.events),
                "states": ray.catalog.states_visited,
                "bounds": ray.catalog.reached_bounds,
                "x0_max_error": float(np.max(np.abs(ray.traces[0] - reference))),
                "unsupported_event_offset_pairs": int(np.count_nonzero(~ray.supported)),
            }
    model = load_blocked_model(ROOT / "data/thick_5layer.csv")
    before = perf_counter()
    single = offset_gather(model, "primaries", angles, np.array([0.0]), dt, n, 20.0, n_fft=8 * n)
    batches = {}
    for far in (1000.0, 4500.0):
        got = offset_gather(
            model, "primaries", angles, np.array([0.0, far]), dt, n, 20.0, n_fft=8 * n
        )
        batches[str(far)] = float(np.max(np.abs(single[0] - got[0])) / np.max(np.abs(single)))
    fine = offset_gather(
        model,
        "primaries",
        np.arange(0.0, 40.0001, 0.0625),
        np.array([0.0]),
        dt,
        n,
        20.0,
        n_fft=8 * n,
    )
    report["slant_sampling"] = {
        "batch_relative_errors": batches,
        "fine_grid_relative_error": float(np.max(np.abs(single - fine)) / np.max(np.abs(fine))),
        "total_seconds": perf_counter() - before,
    }
    water = load_blocked_model(ROOT / "data/water_bottom.csv")
    aperture_errors: dict[str, float] = {}
    report["aperture_constant_relative_errors"] = aperture_errors
    for top in (10.0, 20.0, 40.0):
        p, weights = _aperture_weights(water, np.arange(top + 1), True, 0.0)
        live = p[weights > 0]
        aperture_errors[str(top)] = float(weights.sum() / np.ptp(live) - 1)

    # A deliberately isolated weak multiple: T1's second surface multiple.
    anchor = load_blocked_model(ROOT / "data/single_interface.csv")
    before = perf_counter()
    slant = offset_gather(anchor, "full", angles, offsets, dt, n, 20.0, n_fft=8 * n)
    report["slant_three_trace_seconds"] = perf_counter() - before
    ray = ray_offset_gather(anchor, "full", offsets, dt, n, 20.0)
    early = (times >= 0.05) & (times <= 0.35)
    weak = (times >= 1.4) & (times <= 1.66)
    report["isolated_weak_event_leakage"] = {
        label: {
            "early_peak_over_local_weak_peak": (
                np.max(np.abs(g[:, early]), axis=1) / np.max(np.abs(g[:, weak]), axis=1)
            ).tolist()
        }
        for label, g in (("slant", slant), ("ray", ray.traces))
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), layout="constrained")
    limits = []
    for lo, hi in ((0.05, 0.45), (1.30, 1.66)):
        window = (times >= lo) & (times <= hi)
        limits.append(
            1.05
            * max(float(np.abs(g[:, window]).max() / np.abs(g).max()) for g in (slant, ray.traces))
        )
    for row, (label, gather) in enumerate(
        (
            ("slant: reshaped wavelet / focusing", slant),
            ("ray: unit Ricker / pressure R/T", ray.traces),
        )
    ):
        scale = float(np.max(np.abs(gather)))
        for col, (lo, hi) in enumerate(((0.05, 0.45), (1.30, 1.66))):
            ax = axes[row, col]
            for i, x in enumerate(offsets):
                ax.plot(times, gather[i] / scale, label=f"x={x:g} m", lw=0.9)
            ax.set(
                xlim=(lo, hi),
                ylim=(-limits[col], limits[col]),
                xlabel="time s",
                ylabel="amplitude / backend full peak",
                title=label,
            )
            ax.legend(fontsize=8)
    fig.suptitle(
        "T1 noiseless diagnostic windows: pre-event and isolated weak multiple\n"
        "Identical normalized limits; full-amplitude plots available in offset demo"
    )
    fig.savefig(output / "diagnostic_windows.png", dpi=150)
    (output / "numerical_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
