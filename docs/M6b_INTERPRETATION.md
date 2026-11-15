# M6b: finite spread and multiple identification

## Reproduce

```bash
uv run python scripts/interpretation.py --no-show
```

This is the **diagnostic ray** reference agreed for M6b. The slant default and
its amplitude/wavelet convention are unchanged. T4 is deferred. The command
writes to `output/` (or `--output-dir PATH`):

- `interpretation.md`: budgets, **every** energetic connected region, dominant
  and top-contributing path geometries, finite-spread fits/resolution/fold;
- `interpretation.json`: ordered paths, coherent normal-incidence amplitudes,
  independent forward/inverse moveout errors, fit trajectories and checks;
- `interpretation_{model}_{variant}_ray.npz`: raw panel, gate RMS, fold,
  strong-cell mask and all coincident-group energies on those cells, including
  dominant-event gate shifts and differential timing errors;
- `interpretation_{model}_{stack,trajectories}_ray.png`: small-spread RMS versus
  finite-spread fits on panels and against cumulative spread.

No noise, extra mutes, peak replacements, or synthesis amplitude pruning.
Report generation fails on reflection-bound truncation, failed accounting,
failed independent geometry checks, or unresolved required anchors.

## What is being fitted and what is uncertain?

`mm1d.interpretation.fit_stacking_velocity` minimizes **time** residuals

```
min(T0,V) sum_i [sqrt(T0² + x_i²/V²) - T_exact(x_i)]²
```

with equal trace weights and a free intercept. It does not regress squared
times, weight by amplitude, or maximize semblance. The cumulative trajectory
uses 20/40/60/80/100% of the requested spread. Only supported rays with complete
valid event-centered gates enter. Missing offsets are not extrapolated; the
fit can plateau or differ across variants because their usable samples differ.

The small-spread anchor remains `sqrt(sum(t_leg*v_leg²)/sum(t_leg))` at true T0.
It is not the free-intercept finite-spread velocity. Fitted T0 need not equal
true T0, but the differences here are small.

Resolution is **not a statistical confidence interval**. Freeze the participating
offsets and require every predicted arrival to lie within an explicit timing
budget of the exact Snell arrival. At each scan V, intersect the allowable T0²
intervals; infeasible velocities have no admissible intercept. This includes
the joint T0/V trade-off without a Jacobian approximation. An interval reaching
a scan boundary is labelled censored, not a measured endpoint.

The timing budget is the sum of:

- 1/8 dominant period = **6.25 ms** at 20 Hz (declared phase-resolution budget);
- half a sample = **1 ms**;
- half a T0 scan cell = **4.983 ms (T2)** / **6.645 ms (T3)**;
- half a V cell projected into arrival time over the actual offsets.

This is a conservative, explicit tolerance convention, not an inferred noise
standard deviation. Changing the source, grid or desired phase criterion
changes the budget. The report prints the resulting velocity intervals, stores
joint intercept intervals, and distinguishes event-usable fold from the actual
nearest-fit scan-cell fold. No unrestricted noiseless argmax is required.

## Which ridges count as strong?

Coherence is amplitude normalized. High S alone cannot distinguish a primary
from a tiny deterministic tail. For this report, an energetic/strong cell has
**S ≥0.8**, **gate RMS ≥1% of that variant's maximum gate RMS**, and
**fold ≥13/61**. These are interpretation thresholds, not universal detection
criteria. Raw panels remain unaltered; subthreshold streaks are not hidden.

All selected cells, including single-cell regions, are accounted for. Adjacent
selected cells are grouped using 8-connectivity for reporting, **not** for
asserting one event per connected blob. Each distinct ordering is retained;
equal traversal counts share moveout only after their signed pressure products
are coherently summed. Geometry IDs `G2-4`, for example, count two water-layer
and four sediment-layer traversals, not reflection order.

Each group's isolated gate energy is evaluated on exactly the scanner's
contributing traces, with identical interpolation, validity and slowness tests.
The report lists top groups with energy percentages. Sum of isolated energies
need not equal observed energy: the signed difference is interference cross
energy. A mixed cell has no group contributing ≥80% of summed isolated energy.
Group waveform sums reconstruct the bounded gather exactly in this run. This
is an **accounting check**, not independent validation of ray physics.

Independence comes from analytic constant-layer hyperbolas and Dix limits in
pytest, forward sums over individual path legs versus bracketed inverse Snell
solutions, and M6a's separate omega-p comparisons. Maximum measured forward
check errors over the reported geometries are **4.5e-10 m** / **2.7e-15 s**.

## T2: water, peg-legs, then interbed energy

| Variant | Strong cells | Connected regions | Mixed cells |
|---|---:|---:|---:|
| full | 2313 | 15 | 66 |
| no_surface | 570 | 3 | 0 |
| primaries | 448 | 2 | 0 |

- Primaries are `G2-0` (0.4 s, water bottom) and `G2-2` (0.9 s, sediment base).
- Full-only `G4-0`, `G6-0`, `G8-0`, `G10-0` are the 0.8/1.2/1.6/2.0 s water
  train. Their fitted velocity is **1500 m/s**, with roundoff-level timing
  residuals, at every supported spread. All four have strong cells inside
  their explicit timing envelopes. They are below the same-time primary trend.
- `G4-2` (1.3 s) combines **two** surface peg-leg orderings. RMS is 1897.4 m/s;
  its full-spread fit is **1904.7 m/s**, between water and primary trends.
  `G6-2`, `G8-2`, etc. continue the surface peg-leg sequence. `G6-4`, `G8-4`,
  etc. combine surface and mixed families; they must not be assigned to a
  unique ordering from moveout alone.
- `G2-4` (1.4 s) is an underside reflection at the water bottom with two
  downward reflections at the sediment base. It persists in `no_surface`,
  disappears in `primaries`, and fits at **2183.8 m/s** versus RMS 2181.1 m/s.
  Later high-S interbed streaks in `no_surface` can fall below the energetic
  threshold; their existence does not make them strong separate detections.
- The late full region merges several near-record-end families (2.8–3 s).
  An event with center at 3 s can contribute in-record wavelet tails even
  though no complete event-centered fit gate exists. It is reported without
  inventing an in-record fitted arrival.

The deep primary fits at **2059.8 m/s** versus RMS **2049.4 m/s**, with only
**0.294 ms** max residual, in the primaries panel (61 usable traces). In full,
unsupported neighbour windows reduce it to **42** usable traces and a fit of
**2054.4 m/s**. That difference is support selection, not a new physical RMS.
The shallow primary has only **21** usable traces and a joint-budget velocity
range of roughly **1251–2017 m/s**: a broad apparent ridge is not a precise pick.

Both primary anchors and all four tested surface orders pass the support-aware
ridge-match test. The report does not require every weak or interfering
catalog path to have a separate observable peak.

## T3: coincident families and destructive interference

| Variant | Strong cells | Connected regions | Mixed cells |
|---|---:|---:|---:|
| full | 707 | 20 | 342 |
| no_surface | 557 | 10 | 0 |
| primaries | 330 | 4 | 0 |

The four primary-only anchors pass their timing/support budgets:

| T0 s | RMS m/s | Finite-spread fit m/s | Usable fold | Max residual ms |
|---:|---:|---:|---:|---:|
| 0.5 | 1800.0 | 1800.0 | 16 | <1e-9 |
| 1.0 | 2294.6 | 2318.2 | 60 | 1.474 |
| 1.3 | 2295.8 | 2307.5 | 61 | 0.642 |
| 1.8 | 2511.3 | 2517.7 | 61 | 0.208 |

- At 1.0 s, primary `G2-2-0-0` and surface repeat `G4-0-0-0` share T0 but
  not moveout. Their interference affects the low-velocity surface ridge.
- At 1.5 s, `G6-0-0-0` (pure surface), `G4-2-0-0` (surface peg-legs), and
  `G2-4-0-0` (interbed between interfaces 1 and 2) coincide in T0. The full
  panel is not three independent unconstrained velocity picks.
- `G2-2-4-0` at 1.6 s is the soft-layer reverberation between interfaces
  2 and 3; it is resolved in the no-surface energetic accounting. The soft
  interbed is **not** the only multiple generator in this model.
- At 1.8 s, primary `G2-2-2-2` interferes with surface peg-legs
  `G4-2-2-0` and interbed orderings `G2-4-2-0`. Its nearest full-panel fit
  cell has S≈**0.749**, below the declared strong threshold, with **zero**
  strong cells in its fixed-membership timing envelope. The nearby energetic
  1.741–1.781 s ridge is a displaced/interfering response, not evidence for
  another primary or an incorrect Snell arrival. Primary-only and no-surface
  panels recover strong center matches.
- Later full regions contain overlapping surface/mixed geometries with many
  coincident orderings. Some regions have cross energy fractions below **-2**:
  observed energy is much smaller than summed isolated energies. Even a
  dominant isolated group does not imply negligible interference (no-surface
  regions also show destructive cross terms despite zero “mixed” cells).

Nonhyperbolicity here is generally **sub-millisecond to 1.5 ms** on the usable
moderate spreads, smaller than the declared timing budgets. It cannot explain
all tens-of-ms coherence shifts. Mean gate-center shifts reach about **64 ms**
on energetic cells; differential errors and adjacent fold changes are tabulated
for every region. A common wavelet-flank gate shift, a changing trace ensemble,
and interference must be considered before attributing an apparent streak to
a different event family.

## Limits

This completes the bounded, thick-layer T2/T3 interpretation, not uniqueness of
all noiseless peaks or completeness on arbitrary fine logs. Tiny tails, weak
subthreshold events, low fold, record edges and destructive interference remain
explicit limitations. Coincident orderings cannot be separated by moveout;
family mixtures sharing traversal counts are reported together. No statistical
uncertainty is inferred from noiseless fits. T1–T3 do not demonstrate the
optional fast-layer-over-slow T4 higher-than-primary interbed regime. Slant
aperture/focusing/wavelet effects require separate interpretation, not reuse of
these ray energy or timing thresholds.
