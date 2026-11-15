# mm1d — 1.5D acoustic multiple analysis

Synthetic wavefields from layered P-velocity/density logs: `full`, `no_surface`,
and `primaries`. SI units internally; public angles in degrees. See `PLAN.md`
for validated milestones; [M6b interpretation](docs/M6b_INTERPRETATION.md) explains
finite-spread fits, ridge attribution and their limits.

## Run

Use the existing **uv-managed Python 3.14** environment (AGENTS.md requests 3.15;
runtime migration is deferred).

```bash
uv sync
uv run python scripts/zero_offset.py --no-show
uv run python scripts/angle_gathers.py --no-show
uv run python scripts/offset_gathers.py --backend ray --no-show
uv run python scripts/semblance.py --backend ray --no-show
uv run python scripts/interpretation.py --no-show
uv run python scripts/check_artifacts.py --no-show
```

Figures and numerical reports go to gitignored `output/`. Omit `--no-show` for
interactive viewing. Semblance also writes fold maps, NPZ panels/fold arrays,
and compact event tables. Interpretation writes `output/interpretation.md`/JSON,
cellwise energy-accounting NPZ files, and RMS versus finite-spread figures for
T2/T3. Noise is **off** by default; `--noise-fraction .01`
is an optional, explicitly labelled seeded experiment, not an artifact fix.

## Offset backends (not interchangeable amplitudes)

- **`slant`** remains the default: independent omega-p reflectivity and
  rho-filtered inverse slant stack. The Ricker is reshaped and amplitudes have
  stationary-phase focusing. Finite-aperture smearing remains. Two successive
  waveform refinements must satisfy `2e-4 * per-trace peak + 1e-10`; a 16385-node
  limit raises `QuadratureConvergenceError` if unresolved. DEBUG logging for
  `mm1d.gathers` reports converged node counts/residuals. Cutoff-straddling
  apertures explicitly evaluate `(1-1e-6)*p_critical`, omitting the final 1 ppm
  sliver; untapered near-critical resonances may hit the resource limit.
- **`ray`** is diagnostic: pressure R/T path products, unit-peak fractional-time
  Ricker wavelets, **no geometric spreading or diffraction**. Not a physical
  point/line-source Green function. Files have a distinct `_ray` suffix.
  Zero-offset comparisons use the plane-wave synthetic, not the slant waveform.

```python
import numpy as np
from mm1d.io import load_blocked_model
from mm1d.ray import ray_offset_gather
from mm1d.semblance import semblance_with_fold

model = load_blocked_model("data/water_bottom.csv")
offsets = np.linspace(0.0, 900.0, 61)
result = ray_offset_gather(model, "full", offsets, 0.002, 1500, 20.0)
scan = semblance_with_fold(
    result.traces,
    offsets,
    0.002,
    np.linspace(0.1, 2.8, 271),
    np.linspace(1200.0, 2600.0, 81),
    0.05,
    p_mute_s_m=result.p_max_s_m,
    valid_samples=result.valid_samples,
)
print(result.catalog.reached_bounds, scan.fold.shape)
```

Ray support respects the global subcritical limit and requested surface angle
(default 40°), **without the slant taper**. Unsupported event/offset pairs have
NaN arrival metadata and are not deposited at clipped times. Grey gather regions
mark missing-event wavelet windows; semblance excludes gates touching those
invalid samples. This support mask does not certify infinitely small wavelet
tails or order-truncated catalogs as complete.

Enumeration defaults: zero-offset centers through record end plus a Ricker-tail
margin (`1e-10` per unit wavelet), at most **32 total surface/interface
reflections**, **100000 paths/partial states**; no amplitude pruning. Resource
overflow raises `PathLimitError`; reached reflection/time bounds are reported.
Distinct coincident paths are summed coherently, not discarded. Display catalogs
separately select surface+underside order ≤3 and absolute `|A(0)| ≥0.01`.
Fine logs can require tighter bounds; finite catalogs are not infinite responses.

Noiseless semblance has scale-free coherence ties. RMS markers are small-spread
anchors, not necessarily the unrestricted panel argmax or finite-offset best-fit
velocity; inspect fold alongside coherence. `mm1d.interpretation` fits exact
supported arrivals over cumulative spreads and profiles joint T0/V timing-budget
envelopes (not statistical confidence intervals). `mm1d.ridges` accounts for all
energetic cells using coherent coincident-path groups and exposes interference,
wavelet-flank shifts and support changes. The reproducible report validates six
primary and four water-multiple anchors without requiring unique raw maxima.
See [the report guide and measured findings](docs/M6b_INTERPRETATION.md).

## Checks

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Tests include analytic path signs/multiple trains, subcritical omega-p comparisons,
zero-offset reflectivity cross-checks, quadrature convergence, support/fold, and
weak-event-free windows. `scripts/check_artifacts.py` records reproducible error,
path-count, timing and traced-memory measurements in `output/numerical_checks.json`.
