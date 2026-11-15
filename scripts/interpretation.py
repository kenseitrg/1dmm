"""M6b: reproduce noiseless T2/T3 ray ridge accounting and finite-spread fits.

Writes output/interpretation.md, .json, cell-accounting NPZ files and figures.
All thresholds are explicit; no noise, amplitude pruning or argmax uniqueness.
The slant backend and existing demo defaults are deliberately unchanged.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from mm1d.events import Event, path_amplitudes
from mm1d.gathers import path_offset_time
from mm1d.interpretation import (
    MoveoutResolution,
    finite_spread_trajectory,
    moveout_resolution,
)
from mm1d.io import load_blocked_model
from mm1d.ray import RayGather, event_moveout, ray_offset_gather
from mm1d.reflectivity import ALL_VARIANTS
from mm1d.ridges import attribute_ridge_cells, strong_ridge_cells
from mm1d.semblance import default_t0_grid, default_velocity_grid, semblance_with_fold
from mm1d.viz import semblance_stack_figure

ROOT = Path(__file__).resolve().parents[1]
DT = 0.002
FREQUENCY = 20.0
GATE = 0.05
COHERENCE = 0.8
RELATIVE_RMS = 0.01
N_OFFSETS = 61
MIN_FOLD = 13  # ceil(.2 * 61), same minimum as the scan
RECORDS = {"water_bottom": 3.0, "thick_5layer": 4.0}


def event_usable(ray: RayGather, index: int) -> np.ndarray:
    """Return supported, in-record event-centered gates with all samples valid."""
    half_gate = (max(1, round(GATE / DT)) - 1) * DT / 2
    times = ray.arrival_times_s[index]
    ok = ray.supported[index].copy()
    for row in np.flatnonzero(ok):
        first = int(np.floor((times[row] - half_gate) / DT))
        last = int(np.floor((times[row] + half_gate) / DT)) + 1
        ok[row] = first >= 0 and last < ray.traces.shape[1]
        if ok[row]:
            ok[row] = ray.valid_samples[row, first : last + 1].all()
    return ok


def group_name(events: tuple[Event, ...]) -> str:
    """Stable geometry ID; distinct coherent orderings remain in JSON."""
    return "G" + "-".join(str(n) for n in events[0].traversal_counts)


def resolution_summary(resolution: MoveoutResolution) -> dict:
    """Summarize deterministic joint-intercept envelope, explicitly marking censoring."""
    ok = resolution.feasible
    if not ok.any():
        return {"velocity_interval_m_s": None, "t0_interval_s": None, "censored": False}
    return {
        "velocity_interval_m_s": [
            float(resolution.velocity_m_s[ok].min()),
            float(resolution.velocity_m_s[ok].max()),
        ],
        "t0_interval_s": [
            float(resolution.lower_t0_s[ok].min()),
            float(resolution.upper_t0_s[ok].max()),
        ],
        "censored": bool(ok[0] or ok[-1]),
    }


def interpret_model(model_name: str, record: float, output: Path) -> dict:
    """Generate one model's three panels and complete energetic-region accounting."""
    model = load_blocked_model(ROOT / "data" / f"{model_name}.csv")
    offsets = np.linspace(0, 1100 * model.interface_twt_s[-1], N_OFFSETS)
    t0, velocity = default_t0_grid(record), default_velocity_grid(model)
    spread_limits = offsets[-1] * np.linspace(0.2, 1.0, 5)
    dt0, dv = float(t0[1] - t0[0]), float(velocity[1] - velocity[0])
    panels, overlays, result = {}, {}, {}
    trajectory_fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, variant in zip(axes, ALL_VARIANTS, strict=True):
        ray = ray_offset_gather(model, variant, offsets, DT, round(record / DT), FREQUENCY)
        scan = semblance_with_fold(
            ray.traces,
            offsets,
            DT,
            t0,
            velocity,
            GATE,
            p_mute_s_m=ray.p_max_s_m,
            valid_samples=ray.valid_samples,
        )
        strong = strong_ridge_cells(
            scan,
            coherence_min=COHERENCE,
            relative_rms_min=RELATIVE_RMS,
            min_fold=MIN_FOLD,
        )
        attribution = attribute_ridge_cells(
            model,
            ray,
            offsets,
            DT,
            FREQUENCY,
            t0,
            velocity,
            strong,
            GATE,
        )
        if attribution.reconstruction_error > 2e-12:
            raise AssertionError("coincident-group accounting failed to reconstruct ray gather")
        iv, it = attribution.velocity_index, attribution.time_index
        energies = attribution.group_energy
        total = energies.sum(axis=0)
        dominant = np.argmax(energies, axis=0)
        fractions = np.divide(
            energies.max(axis=0), total, out=np.zeros_like(total), where=total > 0
        )
        cross_fraction = np.divide(
            attribution.observed_energy - total,
            attribution.observed_energy,
            out=np.zeros_like(total),
            where=attribution.observed_energy > 0,
        )
        components, count = label(strong, np.ones((3, 3), dtype=int))
        # A region is an 8-connected selected-cell component, NOT a picked event.
        fold_step = np.zeros_like(scan.fold)
        for axis in (0, 1):
            difference = np.abs(np.diff(scan.fold, axis=axis))
            before = [slice(None), slice(None)]
            after = [slice(None), slice(None)]
            before[axis], after[axis] = slice(None, -1), slice(1, None)
            fold_step[tuple(before)] = np.maximum(fold_step[tuple(before)], difference)
            fold_step[tuple(after)] = np.maximum(fold_step[tuple(after)], difference)
        regions: list[dict] = []
        for region in range(1, count + 1):
            cells = components[iv, it] == region
            ranked = np.argsort(energies[:, cells].sum(axis=1))[::-1]
            top = ranked[:3]
            region_energy = float(energies[:, cells].sum())
            regions.append(
                {
                    "region": region,
                    "cells": int(cells.sum()),
                    "t0_extent_s": [float(t0[it[cells]].min()), float(t0[it[cells]].max())],
                    "velocity_extent_m_s": [
                        float(velocity[iv[cells]].min()),
                        float(velocity[iv[cells]].max()),
                    ],
                    "fold_range": [
                        int(scan.fold[iv[cells], it[cells]].min()),
                        int(scan.fold[iv[cells], it[cells]].max()),
                    ],
                    "dominant_event_fold_range": [
                        int(attribution.dominant_event_fold[cells].min()),
                        int(attribution.dominant_event_fold[cells].max()),
                    ],
                    "top_groups": [
                        group_name(attribution.groups[k])
                        for k in top
                        if energies[k, cells].sum() > 0
                    ],
                    "top_group_percentages": [
                        float(100 * energies[k, cells].sum() / region_energy)
                        for k in top
                        if energies[k, cells].sum() > 0
                    ],
                    "mixed_cells": int(np.count_nonzero(fractions[cells] < 0.8)),
                    "adjacent_fold_change_cells": int(
                        np.count_nonzero(fold_step[iv[cells], it[cells]])
                    ),
                    "center_shift_range_ms": [
                        float(attribution.dominant_center_shift_s[cells].min() * 1000),
                        float(attribution.dominant_center_shift_s[cells].max() * 1000),
                    ],
                    "differential_error_range_ms": [
                        float(attribution.dominant_differential_error_s[cells].min() * 1000),
                        float(attribution.dominant_differential_error_s[cells].max() * 1000),
                    ],
                    "cross_energy_fraction_range": [
                        float(cross_fraction[cells].min()),
                        float(cross_fraction[cells].max()),
                    ],
                }
            )
        # Include every dominant geometry, plus all primaries and pure shallow
        # surface repeats even if interference/low strength hides their ridge.
        inspect = {int(k) for k in dominant}
        for region in regions:
            inspect.update(
                k
                for k, events in enumerate(attribution.groups)
                if group_name(events) in region["top_groups"]
            )
        inspect.update(
            k
            for k, events in enumerate(attribution.groups)
            if events[0].family == "primary"
            or (
                events[0].family == "surface"
                and all(n == 0 for n in events[0].traversal_counts[1:])
            )
        )
        table: list[dict] = []
        fit_curves_t, fit_curves_v, rms_t, rms_v = [], [], [], []
        for k in sorted(inspect):
            events = attribution.groups[k]
            event = events[0]
            index = ray.catalog.events.index(event)
            usable = event_usable(ray, index)
            fits = finite_spread_trajectory(model, event, offsets, spread_limits, usable=usable)
            # Independent forward sum over individual legs, compared to the
            # bracketed inverse geometry at every supported offset.
            p, times, supported = event_moveout(model, event, offsets)
            angles = np.rad2deg(np.arcsin(p[supported] * model.vp_m_s[0]))
            check_x, check_t = path_offset_time(model, event.turning_points, angles)
            x_error = float(np.max(np.abs(check_x - offsets[supported])))
            t_error = float(np.max(np.abs(check_t - times[supported])))
            if x_error > 1e-5 or t_error > 1e-8:
                raise AssertionError("independent forward/inverse moveout check failed")
            fit = fits[-1]
            row: dict = {
                "id": group_name(events),
                "families": sorted({e.family for e in events}),
                "paths": [list(e.turning_points) for e in events],
                "t0_s": event.t0_s,
                "rms_m_s": event.vrms_m_s,
                "coherent_A0": float(path_amplitudes(model, events, np.array([0.0])).sum()),
                "dominant_cells": int(np.count_nonzero(dominant == k)),
                "ray_supported_fold": int(supported.sum()),
                "usable_fold": int(usable.sum()),
                "independent_offset_error_m": x_error,
                "independent_time_error_s": t_error,
                "trajectory": [asdict(f) if f is not None else None for f in fits],
            }
            if fit is not None:
                # Explicit budget, not tuned to a measured pick: 1/8 dominant
                # period + half sample + half time cell + half velocity cell
                # projected to time at the fitted arrivals. Fixed membership.
                velocity_cell_s = float(
                    np.max(offsets[usable] ** 2 / (fit.velocity_m_s**3 * times[usable]) * dv / 2)
                )
                budget = 1 / (8 * FREQUENCY) + DT / 2 + dt0 / 2 + velocity_cell_s
                resolution = moveout_resolution(offsets[usable], times[usable], velocity, budget)
                compatible = (
                    resolution.feasible[:, None]
                    & (t0[None, :] >= resolution.lower_t0_s[:, None])
                    & (t0[None, :] <= resolution.upper_t0_s[:, None])
                )
                fit_iv, fit_it = (
                    int(np.argmin(abs(velocity - fit.velocity_m_s))),
                    int(np.argmin(abs(t0 - fit.t0_s))),
                )
                row.update(
                    {
                        "fit": asdict(fit),
                        "timing_budget_s": budget,
                        "budget_terms_s": {
                            "wavelet": 1 / (8 * FREQUENCY),
                            "sample": DT / 2,
                            "time_cell": dt0 / 2,
                            "velocity_cell": velocity_cell_s,
                        },
                        "resolution": resolution_summary(resolution),
                        "compatible_strong_cells": int(np.count_nonzero(compatible & strong)),
                        "nearest_fit_cell_fold": int(scan.fold[fit_iv, fit_it]),
                        "nearest_fit_cell_coherence": float(scan.panel[fit_iv, fit_it]),
                    }
                )
                # Main panels keep every dominant group, without implying the
                # full bounded catalog would be legible as a plotting overlay.
                if row["dominant_cells"]:
                    curve = [f for f in fits if f is not None]
                    fit_curves_t.extend([f.t0_s for f in curve] + [np.nan])
                    fit_curves_v.extend([f.velocity_m_s for f in curve] + [np.nan])
                    rms_t.append(event.t0_s)
                    rms_v.append(event.vrms_m_s)
                # Separate spread axes: primaries and shallow pinned family.
                if event.family == "primary" or (
                    event.family == "surface"
                    and event.surface_order == 1
                    and all(n == 0 for n in event.traversal_counts[1:])
                ):
                    values = [f.velocity_m_s if f else np.nan for f in fits]
                    (line,) = ax.plot(spread_limits, values, "o-", ms=3, label=row["id"])
                    ax.axhline(event.vrms_m_s, color=line.get_color(), ls="--", lw=0.8)
            table.append(row)
        # Quantitative validation: no demand for a unique unrestricted argmax.
        anchors = [
            r
            for r in table
            if (variant == "primaries" and r["families"] == ["primary"])
            or (model_name == "water_bottom" and variant == "full" and r["families"] == ["surface"])
        ]
        # Pure-water anchors are identified by traversal counts, not coincident
        # deep peg-legs; only centers whose complete near gate is in record.
        if model_name == "water_bottom" and variant == "full":
            anchors = [r for r in anchors if r["id"].endswith("-0") and r["t0_s"] <= 2.0 + 1e-9]
        checks = []
        for row in anchors:
            passed = bool(row.get("compatible_strong_cells", 0) > 0)
            checks.append({"id": row["id"], "supported_ridge_match": passed})
        np.savez_compressed(
            output / f"interpretation_{model_name}_{variant}_ray.npz",
            panel=scan.panel,
            fold=scan.fold,
            gate_rms=scan.gate_rms,
            strong=strong,
            t0_s=t0,
            velocity_m_s=velocity,
            offsets_m=offsets,
            velocity_index=iv,
            time_index=it,
            group_energy=energies,
            observed_energy=attribution.observed_energy,
            dominant_group=dominant,
            cross_energy_fraction=cross_fraction,
            dominant_center_shift_s=attribution.dominant_center_shift_s,
            dominant_differential_error_s=attribution.dominant_differential_error_s,
            dominant_event_fold=attribution.dominant_event_fold,
            group_ids=np.array([group_name(g) for g in attribution.groups]),
        )
        result[variant] = {
            "synthesis_paths": len(ray.catalog.events),
            "bounds": ray.catalog.reached_bounds,
            "strong_cells": int(strong.sum()),
            "regions": regions,
            "groups": table,
            "checks": checks,
            "accounting_error": attribution.reconstruction_error,
            "mixed_cells": int(np.count_nonzero(fractions < 0.8)),
            "energy_threshold_pressure": float(RELATIVE_RMS * scan.gate_rms.max()),
            "low_fold_high_coherence_cells": int(
                np.count_nonzero((scan.panel >= COHERENCE) & (scan.fold < MIN_FOLD))
            ),
            "weak_high_coherence_cells": int(np.count_nonzero((scan.panel >= COHERENCE) & ~strong)),
        }
        panels[variant] = scan.panel
        overlays[variant] = {
            "small-spread RMS": (np.array(rms_t), np.array(rms_v), "wo"),
            "finite-spread fit": (np.array(fit_curves_t), np.array(fit_curves_v), "c.-"),
        }
        ax.set(
            xlabel="requested cumulative spread (m)",
            ylabel="stacking velocity (m/s)",
            title=f"{variant}: solid = fit; dashed = RMS",
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    stack = semblance_stack_figure(
        velocity,
        t0,
        panels,
        overlay_by_variant=overlays,
        suptitle=f"{model_name}: noiseless ray; RMS vs cumulative-spread fits\n"
        "50 ms gate; energetic-region accounting and fold in report/NPZ",
    )
    # Use a legend rather than coincident in-panel text for nearly identical
    # RMS and finite-spread locations. Open circles keep the cyan fit visible.
    for ax in stack.axes[:3]:
        for text in list(ax.texts):
            text.remove()
        ax.lines[0].set_markerfacecolor("none")
        ax.lines[0].set_markersize(8)
        ax.lines[1].set_marker("x")
        ax.lines[1].set_markersize(4)
        ax.legend(
            ax.lines[:2],
            ["small-spread RMS", "finite-spread fit"],
            fontsize=7,
            loc="upper right",
            facecolor="black",
            labelcolor="white",
        )
    stack.savefig(output / f"interpretation_{model_name}_stack_ray.png", dpi=150)
    trajectory_fig.savefig(output / f"interpretation_{model_name}_trajectories_ray.png", dpi=150)
    return {
        "spread_limits_m": spread_limits.tolist(),
        "time_cell_s": dt0,
        "velocity_cell_m_s": dv,
        "variants": result,
    }


def markdown(report: dict) -> str:
    """Render all energetic regions and all dominant moveout geometries, not just examples."""
    lines = [
        "# M6b — noiseless ray interpretation",
        "",
        "Reproduce: `uv run python scripts/interpretation.py --no-show`. "
        "Diagnostic pressure R/T, unit Ricker, no spreading; slant defaults unchanged.",
        "",
        "## Declared budgets and meaning",
        "",
        "- Strong cells: S ≥0.8, gate RMS ≥1% of that variant's maximum, fold ≥13/61. "
        "8-connected regions are accounting units, **not** distinct picked events. "
        "Every selected cell is retained in the NPZ, including single-cell regions.",
        "- Coincident paths sum coherently before energy attribution; all bounded synthesis paths "
        "participate, with no display amplitude/order pruning. Top groups are ranked by isolated "
        "gate energy on the **same contributing traces**. A mixed cell has no group with ≥80% "
        "of summed isolated energies. Signed cross energy = (observed - sum isolated)/observed; "
        "large/negative values expose constructive/destructive interference, not extra events.",
        "- Fits minimize equal-trace arrival-time residuals with free T0 and V over supported "
        "in-record, valid event-centered gates. Trajectories use "
        "20/40/60/80/100% requested spread; "
        "support-limited fits plateau. RMS is the zero-spread limit, NOT a finite-spread pick.",
        "- Resolution intervals profile free T0 over the supplied velocity grid, holding offset "
        "membership fixed. Every arrival must match within 1/8 wavelet period (6.25 ms) + half "
        "sample (1 ms) + half T0 cell + half V cell projected to time. **Not statistical CIs**. "
        "Censored intervals touch a scan edge; an empty envelope means no adequate hyperbola. "
        "Report JSON records every budget term, free-intercept bounds and trajectory fit.",
        "- Primary (primaries panel) and pure-water surface anchors pass if an energetic "
        "supported scan cell lies inside that moveout envelope. No unique raw argmax required. "
        "Independent forward leg-sum checks compare offset/time with bracketed inverse Snell rays; "
        "analytic hyperbolas/Dix/omega-p physics are separately tested in pytest.",
        "- Normalized noiseless tails can tie near S=1 even far from wavelet centers. "
        "Velocity/time extents of an energetic region are NOT uncertainty intervals. "
        "Fold changes between adjacent cells change the measured trace ensemble. "
        "Neither tail ties nor support-boundary streaks imply a new multiple family. "
        "Each region's shift range is mean trial-center minus dominant-event arrival on "
        "actual contributing event-supported traces; differential error removes this common "
        "shift. Thus wavelet flank alignment is visible separately from moveout misalignment. "
        "Region fold columns distinguish all scan contributors from those supporting the "
        "dominant event (dominance may change across cells).",
        "",
    ]
    for model_name, model in report["models"].items():
        lines += [
            f"## {model_name}",
            "",
            f"Grid: ΔT0={model['time_cell_s'] * 1000:.3f} ms, "
            f"ΔV={model['velocity_cell_m_s']:.2f} m/s. "
            f"Requested spread limits: {model['spread_limits_m']} m.",
            "",
        ]
        for variant, data in model["variants"].items():
            lines += [
                f"### {variant}",
                "",
                f"{data['synthesis_paths']} paths; bounds {data['bounds']}. "
                f"{data['strong_cells']} strong cells, {len(data['regions'])} regions; "
                f"{data['mixed_cells']} mixed cells. Gate RMS threshold "
                f"{data['energy_threshold_pressure']:.3g}. "
                f"Reconstruction max error {data['accounting_error']:.2g}.",
                "",
                "#### Every energetic region",
                "",
                "|region|cells|T0 extent s|V extent m/s|scan / event fold|top geometry IDs|mixed|"
                "adjacent fold steps|cross fraction|center shift ms|differential error ms|",
                "|---:|---:|---|---|---|---|---:|---:|---|---|---|",
            ]
            for region in data["regions"]:
                lines.append(
                    f"|{region['region']}|{region['cells']}|"
                    f"{region['t0_extent_s'][0]:.3f}-{region['t0_extent_s'][1]:.3f}|"
                    f"{region['velocity_extent_m_s'][0]:.0f}-{region['velocity_extent_m_s'][1]:.0f}|"
                    f"{region['fold_range']} / {region['dominant_event_fold_range']}|"
                    + ", ".join(
                        f"{name} ({fraction:.1f}%)"
                        for name, fraction in zip(
                            region["top_groups"], region["top_group_percentages"], strict=True
                        )
                    )
                    + "|"
                    f"{region['mixed_cells']}|"
                    f"{region['adjacent_fold_change_cells']}|"
                    f"{region['cross_energy_fraction_range'][0]:+.2f} to "
                    f"{region['cross_energy_fraction_range'][1]:+.2f}|"
                    f"{region['center_shift_range_ms'][0]:+.1f} to "
                    f"{region['center_shift_range_ms'][1]:+.1f}|"
                    f"{region['differential_error_range_ms'][0]:.1f} to "
                    f"{region['differential_error_range_ms'][1]:.1f}|"
                )
            lines += [
                "",
                "#### Dominant / top-contributing geometries + primary / water-family anchors",
                "",
                "IDs are layer traversal counts, not event numbers. JSON lists every ordered "
                "path and every trajectory point. 'match' counts strong cells within the fixed-"
                "membership timing envelope; 'fold' is usable event gates / "
                "actual nearest-fit scan cell.",
                "",
                "|ID|families (path count)|T0 / fit T0 s|RMS / fit V m/s|"
                "resolution V m/s|fold|fit max error ms|match|",
                "|---|---|---|---|---|---|---:|---:|",
            ]
            for row in data["groups"]:
                fit = row.get("fit")
                if fit is None:
                    lines.append(
                        f"|{row['id']}|{','.join(row['families'])} ({len(row['paths'])})|"
                        f"{row['t0_s']:.3f} / —|{row['rms_m_s']:.1f} / —|—|"
                        f"{row['usable_fold']} / —|—|0|"
                    )
                    continue
                interval = row["resolution"]["velocity_interval_m_s"]
                uncertainty = (
                    "empty" if interval is None else f"{interval[0]:.0f}-{interval[1]:.0f}"
                )
                if row["resolution"]["censored"]:
                    uncertainty += " (censored)"
                lines.append(
                    f"|{row['id']}|{','.join(row['families'])} ({len(row['paths'])})|"
                    f"{row['t0_s']:.3f} / {fit['t0_s']:.4f}|"
                    f"{row['rms_m_s']:.1f} / {fit['velocity_m_s']:.1f}|{uncertainty}|"
                    f"{row['usable_fold']} / {row['nearest_fit_cell_fold']}|"
                    f"{fit['max_residual_s'] * 1000:.3f}|{row['compatible_strong_cells']}|"
                )
            lines += [
                "",
                "Anchor checks: "
                + ", ".join(
                    f"{c['id']} {'PASS' if c['supported_ridge_match'] else 'UNRESOLVED'}"
                    for c in data["checks"]
                )
                + ".",
                "",
            ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Generate machine-readable accounting, interpretation report and figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "parameters": {
            "backend": "ray",
            "dt_s": DT,
            "f_dom_hz": FREQUENCY,
            "gate_s": GATE,
            "coherence_min": COHERENCE,
            "relative_gate_rms_min": RELATIVE_RMS,
            "min_fold": MIN_FOLD,
            "n_offsets": N_OFFSETS,
            "max_angle_deg": 40.0,
            "max_reflections": 32,
            "noise_fraction": 0.0,
        },
        "models": {
            name: interpret_model(name, record, args.output_dir) for name, record in RECORDS.items()
        },
    }
    (args.output_dir / "interpretation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "interpretation.md").write_text(markdown(report))
    for name, model in report["models"].items():
        for variant, data in model["variants"].items():
            passed = sum(c["supported_ridge_match"] for c in data["checks"])
            print(
                f"{name}/{variant}: {data['strong_cells']} strong cells; "
                f"{len(data['regions'])} regions; anchors {passed}/{len(data['checks'])}"
            )
    print(f"saved {args.output_dir / 'interpretation.md'} and JSON/NPZ/figures")
    if any(
        not check["supported_ridge_match"]
        for model in report["models"].values()
        for variant in model["variants"].values()
        for check in variant["checks"]
    ):
        raise RuntimeError("unresolved required anchors; inspect interpretation report")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
