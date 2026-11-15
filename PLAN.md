# PLAN.md — mm1d Implementation Plan

Roadmap for `mm1d`. AGENTS.md defines conventions; this file defines *what gets
built, in which order, and when each milestone is done*. Update it as milestones
complete or scope changes (record deviations in the changelog at the bottom).

## Guiding principles

1. **Staged pipeline** (AGENTS.md): `logs → model → reflectivity → synthetics →
   gathers → semblance → events/viz` — every milestone ends with a runnable,
   tested stage, not a library-only increment.
2. **Analytic validation before eyeballs, eyeballs before regression baselines**:
   every physics component is checked against closed-form results on the
   canonical models below; figures must be interpretable by eye before any
   "looks right" snapshot baselines are recorded.
3. **Thick layers for tests**: layer TWT ≥ ~3 dominant wavelet periods, so events
   are separated and visually countable.
4. **Visual verification**: each milestone that produces a figure closes with a
   figure check — by the current model if it can view images, otherwise via the
   `glm-5.3-flash` delegation command in AGENTS.md.
5. **V1 simplifications**: acoustic P only (no converted waves), Ricker source
   wavelet. Everything is structured so elasticity / other wavelets / Q slot in
   later without redesign.

## Canonical test models (`data/`, SI units)

Reference source: 20 Hz Ricker (dominant period 50 ms; wavelet length ≈ 150 ms).

### T1 — `single_interface.csv` (analytic anchor)

| Layer | h (m) | vp (m/s) | rho (kg/m³) | TWT (s) |
|---|---|---|---|---|
| 1 | 500 | 2000 | 2000 | 0.5 |
| half-space | ∞ | 3000 | 2500 | — |

R₁ = (3000·2500 − 2000·2000)/(3000·2500 + 2000·2000) ≈ 0.304.
Full wavefield (free-surface reflection **R_s = −1**, pressure-release — see
AGENTS.md sign conventions): arrivals at tₙ = (n+1)·2h/v = (n+1)·0.5 s with
amplitudes R₁·(R_s·R₁)ⁿ = R₁·(−R₁)ⁿ — alternating polarity: +0.304, −0.093,
+0.028, −0.009
(verified against the exact packet recursion at planning time). No transmission
losses (path never crosses the interface).
`primaries` variant: single event, amplitude R₁. `no_surface`: identical to
`primaries` here (no interbed possible). Also the angle-dependent anchor:
amplitude(θ) from the acoustic coefficient with this impedance pair.

### T2 — `water_bottom.csv` (surface-multiple demonstrator)

| Layer | h (m) | vp (m/s) | rho (kg/m³) | TWT (s) |
|---|---|---|---|---|
| 1 (water) | 300 | 1500 | 1000 | 0.4 |
| 2 (sediment) | 600 | 2400 | 2100 | 0.5 |
| half-space | ∞ | 3000 | 2300 | — |

R₁ ≈ 0.541 (water bottom, strong), R₂ ≈ 0.156.
Primary events at 0.4 s and 0.9 s. Water-bottom multiples at 0.8, 1.2, 1.6, 2.0 s
(period 0.4 s), amplitudes R₁·(−R₁)ⁿ = +0.54, −0.29, +0.16, −0.09 — slowly
decaying, **alternating polarity** (the textbook water-layer signature and a
direct visual test that `R_s = −1` is wired correctly); peg-legs (water-bottom
bounce + sediment internal path) around 1.3–2.1 s. The `full − no_surface`
difference section is a clean water-layer multiple train. T2 is the primary
workhorse for M2–M5.

### T3 — `thick_5layer.csv` (interbed-multiple demonstrator)

| Layer | h (m) | vp (m/s) | rho (kg/m³) | TWT (s) |
|---|---|---|---|---|
| 1 | 450 | 1800 | 2000 | 0.5 |
| 2 | 675 | 2700 | 2300 | 0.5 |
| 3 (soft interbed) | 345 | 2300 | 2150 | 0.3 |
| 4 | 750 | 3000 | 2300 | 0.5 |
| half-space | ∞ | 3600 | 2500 | — |

R₁ ≈ +0.266, R₂ ≈ −0.113, R₃ ≈ +0.165, R₄ ≈ +0.132 (recompute exactly in tests).
Primaries at 0.5, 1.0, 1.3, 1.8 s. Layer 3 is the interbed generator: internal
reverberations between interfaces 2 and 3 with 0.3 s period; peg-legs combining
surface and interbed bounces. T3 is the workhorse for M6 (event catalog).

(Reflectivity values above are hand-computed approximations quoted for design
only — tests must compute them from the model, not hard-code these numbers.)

## Numerical starting defaults (tune per model, document in script headers)

| Parameter | Default | Rationale |
|---|---|---|
| dt | 2 ms | ≤ 1/25 of dominant period, smooth Ricker sampling |
| record length T | 3 s (T1/T2), 4 s (T3) | ≥ 2× deepest primary + multiple train |
| Ricker f_dom | 20 Hz | thick-layer separation requirement |
| Angle range | 0–40°, Δθ = 1° | below critical for T2/T3 top interfaces |
| Frequency grid | Δf = 1/T up to Nyquist | FFT-consistent with time grid |
| Slowness grid | from θ grid via Snell: p = sinθ/v per layer | evanescent p > 1/v_max zeroed |

## Milestones

### M0 — Project scaffold ✅ (complete 2026-11-06)
**Goal**: empty-but-green project skeleton.
- `uv init` (Python 3.14), `src/mm1d` layout, `pyproject.toml` with ruff (lint +
  format config), ty, pytest config; `.gitignore` (`.venv/`, `output/`,
  `__pycache__/`); `data/`, `scripts/`, `tests/`, `output/`.
- Stub modules per AGENTS.md layout; `uv run pytest` passes with one smoke test.
**Done when**: `uv sync && uv run ruff check . && uv run ruff format --check . &&
uv run ty check && uv run pytest` all green; `uv.lock` committed.

### M1 — Model building (`io.py`, `model.py`, `units.py`) ✅ (complete 2026-11-06)
**Goal**: logs → validated layered model → reflection coefficients.
- CSV/TXT loaders (single 3-column file; or two 2-column files merged on depth
  with tolerance); validation: monotonic depth, positive v/ρ, NaN handling with
  clear errors.
- `LayeredModel` dataclass (per-layer h, vp, rho + interface R/T at normal
  incidence and angle-dependent acoustic R(θ), T(θ); energy check R²+T²=1).
- Resampling/blocking from irregular depth log to uniform-TWT grid (thin-zone
  averaging for later fine-grid runs).
- Canonical models written to `data/` + pytest fixtures.
**Done when**: unit tests for loaders/errors pass; R/T coefficients match
hand-computed values for T1–T3 interfaces; `R²+T²=1` for all θ in grid.

### M2 — Zero-offset synthetics, three variants (`wavelet.py`, `reflectivity.py` normal-incidence path) ✅ (complete 2026-11-06)
**Goal**: `1dmm_ud.ipynb` logic, productized.
- Ricker wavelet (time + frequency domain, consistent normalization).
- Normal-incidence recursion for the three variants (switch semantics per
  AGENTS.md table); free-surface convention **R_s = −1** (pressure-release),
  single-sourced in code and validated by the alternating-polarity multiple train
  (resolved at planning time — notebook's `+1` was a toy simplification).
- Script `scripts/zero_offset.py`: three stacked traces for T2 and T3 →
  `output/zero_offset_{model}_{variant}.png`.
**Done when**:
- T1: `full` peaks at (n+1)·2h/v = (n+1)·0.5 s match R₁·(R_s·R₁)ⁿ within 1e-6
  (impulse response, no wavelet; times corrected from the original “1.0 s”
  anchor — see changelog); `primaries` single arrival R₁.
- T2: water-bottom multiple amplitudes follow the analytic series including
  transmission terms; `full − no_surface` and `no_surface − primaries` vanish
  above 0.35 s (before first return).
- Figure check (M2 figure): events countable by eye, variants visibly differ.

### M3 — ω–p reflectivity engine → angle gathers (`reflectivity.py` full) ✅ (complete 2026-11-08)
**Goal**: Müller recursion per (ω, p); angle-domain gathers for three variants.
- Recursive reflectivity/transmissivity of the stack (compound-matrix style,
  stable for all subcritical p); evanescent cutoff; free-surface receiver term.
- Variant switches reused from M2 (same enum, one implementation of semantics).
- `gathers.py` start: angle gather assembly θ ∈ [0°, 40°] for the three variants.
- Script `scripts/angle_gathers.py` → `output/angle_gather_{model}_{variant}.png`.
**Done when**:
- θ = 0 column of every angle gather equals the M2 zero-offset trace (≤ 1e-8
  relative, impulse domain).
- T1 angle gather: arrival times/angles match 2·h·cos… plane-ray predictions
  (x(θ), T(θ) closed forms); amplitude trend vs θ matches acoustic R(θ).
- Recursion stable across full p grid (no overflow warnings at critical p).

### M4 — Angle → offset conversion (`gathers.py`) ✅ (complete 2026-11-09)
**Goal**: offset gathers via Snell-law ray tracing.
- Per-layer ray parameter: p = sinθ_i/v_i constant; x = Σ hᵢ tanθᵢ, T = Σ
  hᵢ/(vᵢ cosθᵢ) over the event's legs; event-agnostic gather transform
  (map θ→x per trace, resample amplitude to uniform offset grid)
  — implemented as the exact-inverse ρ-filtered slant stack plus the
  ray-tracing layer; see changelog 2026-11-09 for the design decision.
- Script `scripts/offset_gathers.py` → `output/offset_gather_{model}_{variant}.png`.
**Done when**:
- At x = 0 the legacy slant stack preserves event families and approximate
  interior arrival times, **not** the M2 waveform/amplitudes (rho-filter and
  focusing differ). The additive diagnostic ray backend below provides the
  R/T-only Ricker comparison. Offset of T1 primary vs θ matches
  closed-form 2h·tanθ; round-trip angle↔offset consistent within interpolation
  error; near/far offsets of the T2 water-bottom multiple follow the theoretical
  moveout `T(x) = sqrt(T0^2 + x^2/v_w^2)` — V_stk pinned at the water velocity,
  i.e. **lower** apparent velocity (steeper moveout) than a primary at the same
  T0 would have (V_rms(0.8 s) ≈ 2001 m/s): the moveout separability behind
  radon demultiple. "Flatter than primary" (higher V_stk at equal T0) is an
  *interbed* phenomenon requiring a fast reverberating layer over a slower
  deeper section — not producible by T1–T3 (expectation corrected 2026-11-09,
  see changelog).

### M4.2 — P0 numerical corrections ✅ (HANDOVER follow-up, 2026-11-15)

- [x] Constant-integrand regression tests first: short subcritical apertures end
  at the requested endpoint; nonuniform/mute/taper cases retain nonnegative
  weights and exclude muted nodes.
- [x] Cutoff-straddling policy: explicitly evaluate `(1-1e-6)*p_critical` and
  integrate to it, omitting the last 1 ppm sliver. Never extend an endpoint
  weight across an unevaluated segment or evaluate the singular critical point.
- [x] Kernel sampling is only an initial lower bound. Require **two successive
  output waveform refinements** within `2e-4 * each trace peak + 1e-10`, including
  x=0, all variants, T1–T3, Ricker/impulse bandwidths and repeated fine grids.
- [x] Stream response in 32-p blocks; no p×frequency×offset temporary. The
  16385-node guard raises `QuadratureConvergenceError` with residual/budget;
  DEBUG logs expose successful counts/residuals. Untapered near-critical full
  responses can fail this bound rather than silently returning aliased data.
- [x] Original T3 primary reproducer: x=0 vs batches `[0,1000]`/`[0,4500]`
  relative max errors **2.97e-8 / 3.66e-8**, fine 0.0625° reference **3.71e-8**
  (previous single-offset error ~9.76e-2). Untapered 10°/20° aperture integral
  errors reduced from +93.97%/+23.10% to floating-point precision.

Finite-aperture distortion is separate: refinement does NOT promise clean
Ricker waveforms or exact measured peaks near aperture edges.

### M5 — Semblance (`semblance.py`) ✅ (original checkpoint 2026-11-10; support follow-up 2026-11-15)
**Goal**: Neidell-Taner semblance panels per variant.
- Sliding-window semblance over (T₀, V_stk) on hyperbolic moveout; shared for
  all three variants; parameter ranges auto-scaled from gather extent.
- Script `scripts/semblance.py` → `output/semblance_{model}_{variant}.png`.
**Done when**: on T2 `primaries` panel, maxima at (0.4 s, V_rms₁) and
(0.9 s, V_rms₂) within one grid cell of Dix values; on `full` panel the
water-bottom multiple train forms a ridge pinned at the water velocity
(V_stk = v_w at every order) — below the same-T₀ primary Dix trend, the
free-surface-multiple signature — while interbed/peg-leg ridges sit between
that trend and the pinned surface-multiple ridges. Higher-than-primary
apparent velocities belong to fast-layer interbed generators (none in
T1–T3; candidate T4 model for M6) — expectation corrected 2026-11-09,
see changelog.

The original completion budgets above refer to the seeded legacy checkpoint.
The HANDOVER follow-up keeps raw gathers noiseless by default, reports fold,
and treats unconstrained local argmax shifts separately from fixed-time RMS
anchors (see M6a); it does not claim one-cell picks on all raw ray panels.

### M6a — Bounded paths + diagnostic synthesis ✅ (brought forward, 2026-11-15)

**Decision confirmed with user:** additive `mm1d.ray.ray_offset_gather`, using
pressure R/T path products, unit-peak Ricker, no spreading; existing
`offset_gather` and its default/amplitudes are unchanged.

- [x] Immutable `Event` and `EventCatalog`: alternating surface/interface turns,
  traversal counts, family, zero-offset time, RMS velocity, directional R/T
  interactions. Surface reflection is -1; underside reflection is -R. No surface
  factor at source/receiver. Coefficients use a shared horizontal p, not a
  surface angle incorrectly applied at a deeper interface.
- [x] Enumerate by recording time + per-wavelet tail margin; default **32 total
  interface + surface reflections**, receiver excluded; **100000 completed or
  pending/visited states** resource guard. Reached bounds reported; overflow
  raises `PathLimitError`. No synthesis amplitude pruning. Surface order counts
  surface bounces; internal order counts underside bounces. Human display uses
  order ≤3 and absolute |A(0)| ≥0.01, explicitly heuristic and separate.
- [x] Keep all coincident orderings; reuse moveout for equal traversal counts
  only AFTER coherently summing distinct path coefficient contributions.
- [x] Bracketed Snell solutions and direct fractional-time Ricker evaluation;
  global subcritical limit plus requested angle aperture, no slant taper.
  Unsupported events have NaN arrival metadata, support flags, and invalid
  sample windows; never clipped to the aperture or replaced by another backend.
- [x] Independent anchors: T1 geometric train/hyperbolas and exact Ricker samples;
  T2 water-velocity multiples; T3 underside/mixed paths and coincident peg-legs;
  all variants against padded normal-incidence AND selected nonzero-p omega-p
  time windows (test budget 2e-9 absolute). Primary frequency sums are numerical
  precision; finite multiple frequency sums converge but are not exact infinite
  responses (T2 full reflection orders 3/7/17: 0.432/0.177/0.0191 max errors).
- [x] T3 4 s synthesis reflection-limit convergence 5/11/17/23/32: max waveform
  differences vs 32 are 1.02e-2 / 2.16e-4 / 3.49e-17 / 0 / 0. The default
  catalog has **3588 paths / 11914 states**, no reflection-bound hit. This is
  evidence for this window/model, not a completeness guarantee for fine logs.
- [x] Isolated weak T1 multiple: near/mid/far event-free windows tested against
  analytic wavelet tails relative to the weak multiple, not only gather peak;
  no added noise, forced x=0 replacement, stronger clipping or arbitrary mutes.

**Integration / measured limits:**
- `scripts/offset_gathers.py --backend ray` writes distinct `_ray` files;
  shared scales across variants, appropriate-family overlays, grey missing-event
  windows. `scripts/check_artifacts.py` writes unclipped normalized weak/near
  waveform comparisons and `output/numerical_checks.json`.
- `semblance_with_fold` returns actual contributing counts; accepts bool sample
  validity. Ray scans use actual untapered support; legacy scans retain their
  taper-clean bound. Existing ndarray-only `semblance_panel` API is preserved.
- `scripts/semblance.py --backend ray` saves noiseless panels + fold PNG/NPZ and
  display event tables; family RMS markers are variant-appropriate. Seeded noise
  is optional (`--noise-fraction .01`) and explicitly labelled.
- **Reassessed bias:** clean gathers do NOT make raw noiseless free local argmax
  unique. T2 primary picks shift -95.7/-66.4 m/s with +48.5/+46.8 ms gate shifts;
  T3 picks range about -84..+80 m/s. These are coherence ties/support and finite
  spread effects, not proof of incorrect ray timing. A separate fixed-true-time
  T1 velocity anchor is within 5 m/s; independent hand-built hyperbola tests stay.
  The old one-cell deep-interface criterion is not asserted for unrestricted
  noiseless argmax. Small-spread RMS remains distinct from finite-offset fitting.
- At standard full offset demo spread T2 primaries have 49 unsupported
  event/offset pairs of 122, T3 113/244; grey is missing support, not measured
  zero energy. Validity is per wavelet window; missing per-event tails are bounded
  only by the declared tolerance, not a proven whole infinite-series norm.
- 4 s / three offsets `[0,200,600]`: x=0 max differences vs 32× padded plane-wave
  traces ≤1.9e-15 across T1–T3/variants. Instrumented T3 full took ~1.82 s with
  ~2.43 MB Python-traced peak allocation (not process RSS); timings vary. Slant
  single/batch/fine reproducer total ~7.58 s; T1 full three-offset ~0.65 s.

### M6b — Interpretation & finite-spread trajectories (remaining)

**Goal:** complete multiple identification, not just a finite synthesis catalog.
- [ ] Add finite-offset best-fit stacking trajectories separately from the
  already plotted (T₀, small-spread RMS) anchors; print uncertainties/fold.
- [ ] Attribute every strong T2/T3 semblance ridge and document noiseless ties,
  interference and scan-cell support changes. Current family markers/tables are
  aids, not a claim that every visible coherence streak is a distinct event.
- [ ] Validate primary/surface ridge matches with explicit support/resolution
  budgets; do not require unique raw argmax on wavelet tails or low-fold cells.
- [ ] A fast-layer-over-slow T4 remains optional for higher-than-primary
  interbed stacking velocities; not demonstrated by T1–T3.

**Done when:** the interpretation report explains the supported strong ridges
with independent moveout checks and separates resolution/nonhyperbolicity from
multiple-family identification. Dedicated `event_catalog.py` is deferred:
current display tables/anchors are integrated into `scripts/semblance.py`.

### M7 — Polish & real-log workflow
**Goal**: first usable tool.
- End-to-end script for arbitrary user CSV pair (argparse: log files, wavelet
  frequency, θ range, semblance ranges); README with 10-minute usage guide;
  final pass of ruff/ty; refresh AGENTS.md/PLAN.md if conventions drifted.
- Optional stretch (only if time allows): auto-picked "multiple horizons" report
  ranking generators by total multiple energy at target time.
**Done when**: a colleague can run one command on their own logs and get the full
figure set; all quality gates green.

## Risks / open questions (resolve before or during the named milestone)

- **Free-surface sign convention** (R_s = +1 vs −1, displacement vs pressure):
  **RESOLVED at planning** — R_s = −1 (pressure-release), consistent with the
  standard interface coefficient convention; M2 tests require the alternating
  multiple train `R₁·(−R₁)ⁿ` (and its absence in `no_surface`). A consistency test
  that `+1` wiring yields the all-positive (non-physical) train guards against
  regressions of the convention.
- **Amplitude fidelity at angle**: plane-wave gathers have no geometric
  spreading; decide whether to include obliquity/transmission-loss-only
  normalization (M3) and state it on every angle-gather figure.
- **Interpolation artifacts in angle→offset** (stretch at large θ): choose
  interpolation order + mute beyond max offset (M4).
- **Wavelet side lobes** vs "interpretable by eye" criterion on thick models:
  acceptable at 20 Hz with ≥ 250 ms TWT layers; revisit if TWT reduced.
- **Runtime documentation discrepancy**: AGENTS.md requests Python 3.15 while
  `.python-version`, pyproject and ty use 3.14. Existing uv environment retained;
  reconcile in a dedicated runtime migration, not these physics fixes.
- **Semblance bias for interfering events** in `full` gathers: compare against
  variant-separated panels — this is a feature of the tool, not a bug; make sure
  M6 report phrasing reflects it.

## Changelog

- 2026-11-15: **HANDOVER implementation** — completed P0 aperture quadrature and
  response-aware convergence; brought M6a bounded paths forward and added the
  user-confirmed R/T-only diagnostic ray backend without changing slant defaults.
  Added support-aware noiseless semblance/fold, backend-labelled demos, separate
  display tables/RMS anchors and reproducible weak-event/numerical checks. Updated
  README and corrected overbroad old timing/zero-offset/smearing statements.
  M6b finite-spread interpretation remains explicitly open rather than claiming
  every ridge is explained. Quality gates: ruff check, ty check, **294 pytest
  tests passed** (baseline 199); changed-file format check and diff check pass. Figure
  sets for both backends generated with --no-show and inspected directly.
  T2 ray panels show the water-bottom multiple ridge near 1500 m/s in full only;
  T3 ray primaries show four arrivals/Dix anchors, with no surface-family overlay;
  grey far-offset wavelet windows correctly expose unsupported rays. Normalized
  weak-event plots retain visible slant pre-event content but no unexplained ray
  energy outside analytic tails. T1 0.05–0.35 s leakage relative to the isolated
  weak second surface-multiple peak at offsets 0/200/600 m: slant
  0.125/0.528/0.196; ray ≤4.8e-37 (analytic Ricker tails). These are different
  amplitude/wavelet conventions, not a samplewise equality test. Fold panels
  expose support variation; noiseless
  coherence streaks/late-tail ties remain visible, not hidden with noise or mutes.
  Python 3.14/3.15 documentation discrepancy recorded without runtime migration.

- 2026-11-10: **M5 figure rework (user request)** — semblance panels rotated to
  the seismic-section convention: zero-offset time is now the *vertical* axis
  (0 at top, increasing downward, as the M3/M4 gather panels), stacking
  velocity horizontal — so variant panels compare side by side in *columns*
  (`semblance_stack_figure`: one column per variant, ``full`` leftmost;
  single-panel figures rotated to match).  API: `plot_semblance` /
  `semblance_stack_figure` now take ``(v_grid, t0_grid, panel(s))`` in
  (x, y, data) order like the other gather plotters.  Colorbar placement
  fixed: stack figures use constrained layout so the figure-level colorbar's
  space is reserved (tight_layout + multi-parent `fig.colorbar` dropped it on
  top of the last panel); single figures likewise; long titles wrapped to two
  lines with a compact scan note (suptitle shortened — the full note lives in
  panel titles).  glm-5.3-flash verified on water_bottom (single + stack) and
  thick_5layer stack: columns ordered full/no_surface/primaries, time down,
  velocity right, colorbar outside all panels, no clipped titles/labels, Dix
  curve threading the four T3 primary blobs.
- 2026-11-10: **M5 complete** — `semblance.py`: `semblance_panel` (the Neidell-Taner
  coherency `S = Σ_j (Σ_i d_ij)² / (N Σ_ij d_ij²)` on a centered gate over hyperbolic
  moveout, linear-interpolation sampling, per-cell exclusion of traces whose gate
  leaves the record or whose hyperbola-local slowness `p = x/(V²t)` reaches the scan
  bound), `default_t0_grid` / `default_velocity_grid` (auto-scaled:
  `[0.8·min(vp), 1.25·max Dix]`), `dix_rms_velocities` (M6's path-leg formula for
  primaries), `scan_slowness_bound` (= `(1 − taper)·min(mute, p(θ_max))`).
  `viz.plot_semblance` / `semblance_stack_figure` (magma panels, Dix-curve +
  velocity-pin overlays with per-curve styles); `scripts/semblance.py` →
  `output/semblance_{model}_{variant}.png` + `_stack.png`; 17 new tests (199 total).
  Design decisions & findings:
  (a) **Scan aperture**: `x ≤ 1.1 km/s × T0(deepest primary)` — Dix comparability needs
  small spreads (at the full M4 aperture the non-hyperbolic moveout drags picks
  +25-35 m/s above Dix); at this aperture deep-interface picks track Dix within
  about one velocity cell.
  (b) **Taper-clean slowness bound**: beyond the synthesis taper the M4-documented
  event tails are attenuated (up to 25x) and phase-shifted (~10 ms early) — summing
  them poisons cells; the bound excludes them per cell, event-agnostically, via the
  hyperbola's local slowness (for a pure-layer event that *is* its stationary ray,
  so e.g. T2's P1 scans only x ≲ 280 m).
  (c) **Scale-free coherence ties**: S is amplitude-normalized per cell, so on
  noise-free deterministic gathers every gate placement inside an event's wavelet
  extent ties near S ≈ 1 and argmax is ill-posed — a seeded 1 % incoherent noise
  floor (caller side, field-data reality) makes maxima well-posed at alignment.
  (d) **Shallow-interface resolution floor (deviation from the one-cell anchor)**:
  the *shallowest* interface of each model is mute-limited to x/T0 ≈ 0.7 at the
  taper-clean bound; with a 20 Hz wavelet its V_stk is resolved only to ~±100 m/s
  (a uniform gate shift inside the broad wavelet is a near-symmetry of S).  Measured
  picks: T2 P1 −79 m/s (t0 +19 ms), T1 P1 −85, T3 P1 −63 — the bias is deterministic
  (unchanged under 0-5 % noise and under gather roughening/balancing; tested and
  rejected: d/dt, d²/dt², boxcar-detrend, gate-local AGC, trace-peak balancing —
  none move the shallow picks).  All *deeper* interfaces pin to Dix within ≈1 cell
  (T2 P2 +18.7 m/s ≈ 1.1 cells at 17 m/s spacing; T3 P2/P3/P4 ≤ 0.6 cells).  Tests
  assert the one-cell anchor for deep interfaces and the ±100 m/s floor for the
  shallowest.
  (e) **Acceptance physics verified**: T2 `full` water-bottom train (orders 1-4)
  pinned within ±28 m/s of v_w at every order, below the same-T0 Dix trend;
  peg12 picked within 0.5 m/s of its path-leg V_stk (1897.4) between pin and trend;
  T1 primary + first-order multiple both at v₁; T3 four primaries with the Dix curve
  threading the blobs.  Figures verified via glm-5.3-flash (pinned ridge visible
  and tracked by the dotted pin line, Dix curve through primary blobs, primaries
  panel free of multiple energy, `no_surface` lacks the pinned ridge); one figure
  nit fixed after review (distinct line styles + label placement for pin vs Dix
  curve).  Bug found & fixed during the scan-aperture work: an initial units slip
  (x_max = 1.1·T0 in m/s not applied) produced 0.99 m offsets and saturated S ≈ 1
  panels — caught by the hand-built-hyperbola anchors.
- 2026-11-09: **M4.1 — offset-gather artifact control** (user request:
  "do not like wrap-around artifacts").  Diagnosis first: literal FFT
  wrap-around was never the visible issue (8x record extension measured
  < 3e-5 of peak); the visible artifacts were (a) acausal radiation of the
  hard slowness-aperture truncation — worst just below the post-critical
  mute where |U| grows toward resonance (~3 % of peak, near-mid offsets) —
  and (b) far-offset quadrature aliasing of the 1 deg p-grid (~3-5 %).  A
  causal one-sided (Sommerfeld/Weyl) kernel was prototyped and REJECTED:
  it removes the acausal smear but leaves ±10-20 % event-amplitude ripple
  (the uncancelled p = 0 endpoint term interferes with every event) — the
  mirrored cos kernel stays (amplitude continuity, exact-pair structure).
  Shipped: `DEFAULT_APERTURE_TAPER = 0.15` Tukey (cos²) taper over the end
  of the (mute-clipped) aperture — parameter `taper_fraction` on
  `offset_gather`/`offset_impulse_gather` — plus internal anti-aliasing
  refinement of the synthesis p-grid to `delta_p <= 1/(4 f_max x_max)`
  (`_refine_factor`/`_refine_angles`, cap 16x; f_max = 3·f_dom for Ricker
  gathers, Nyquist for impulse).  Result (T2 full, measured): acausal
  pre-event energy 3.3 % → 1.2 % of peak, far-offset pre-event 3.2 % →
  1.2 %, stationary-phase anchors unchanged for θ ≤ 15° (≤ 0.4 %; the
  20° anchor sits in the taper-skirt interference band, ≤ 10 %), event
  interior picks within the tested timing tolerance (not a universal
  edge/overlap peak guarantee), round trip unchanged (corr ≈ 0.96, scale ≈
  0.88-0.95).  The residual near-offset pre-event ramp (a few % of peak,
  deepest events worst) was originally attributed entirely to finite-aperture
  content. **Superseded by M4.2:** quadrature was not response-aware, so that
  blanket attribution was unjustified. Converged synthesis still has aperture
  smearing; the additive independently validated ray backend now supplies clean
  diagnostic Ricker deposition (the earlier rejection has been revisited).  Tests: grid-equivalence mute test replaced (the
  near-mute resonant sliver made it meaningless — 0-40° vs 0-29° differ by
  ~5-8 % through different quadratures of a near-singular integrand) by
  exact weights-level mute exclusion + behavioral causality/taper tests +
  refinement-convergence test + helper unit tests (35 offset tests, 182
  total).  Figures regenerated; glm-5.3-flash verification: pre-event zone
  visually clean on T2 (`blank white above P1`) and T3 (four primaries on
  their dashed curves; faint near-offset smudge = documented ramp).
- 2026-11-09: **M4 complete** — `gathers.py` grew two layers. **Snell ray
  tracing**: `layer_angles` (refraction per layer, NaN evanescent),
  `path_offset_time` (arbitrary bounce paths as turning-point sequences over
  `{SURFACE = −1, 0..n_interfaces−1}`; legs crossed → `x += h tanθ`,
  `T += h/(v cosθ)`; reused by M6), `primary_offset_times`, and
  `default_offset_grid` (uniform grid to the deepest primary's offset at the
  last unmuted angle). **Offset-gather synthesis**:
  `offset_impulse_gather`/`offset_gather` — the event-agnostic
  ρ-filtered inverse slant stack evaluated in the frequency domain on the
  *mirrored* slowness grid, `D(x,ω) = (|ω|/π)Σ_j w_j U_j cos(ω p_j x)`
  (exact because the engine depends on p² only; pinned by a
  `_stack_terms`-evenness test); trapezoid weights are *clipped* at the
  critical mute (0–40° synthesis is then bit-identical to the 0–29° one);
  `slant_stack` is the one-sided forward τ-p transform for validation.
  `viz.plot_offset_gather`/`offset_gather_stack_figure`;
  `scripts/offset_gathers.py` → `output/offset_gather_{model}_{variant}.png`
  + `_stack.png` for T1–T3 with primary/multiple moveout overlays.
  Deviations/design decisions: (a) **"map θ→x per trace" was not
  implementable as written** — offset is event-dependent (`τ'(p) = −x(p)`:
  at one θ the T2 water-bottom multiple and P2 go to different offsets), so
  the event-agnostic transform is the exact-inverse synthesis; every event
  of the engine (any multiple family) lands on its own `(x(p*), T(p*))`
  with no catalog, keeping M5/M6 semblance-vs-catalog non-circular (user
  decision: "both transforms" — ray tracing for overlays/tests/M6,
  slant stack for the gathers). (b) **Amplitude convention**: 2-D
  line-source plane-wave synthesis — the |ω| ρ-filter reshapes the wavelet
  by √ω and event peaks scale as `A(p*)/√(2π|dx/dp|)`, continuous through
  x = 0; anchored to 3 % on T1 for θ ≤ 20° (a first attempt doubling only
  the `e^{−iωpx}` branch gave a spurious factor-2 jump at x = 0; the
  mirrored cos kernel fixed it). Stationary rays within ~a Fresnel zone of
  the aperture edge or mute ring up +10–20 % then fade (edge diffraction;
  measured peaks can shift near the edge; this is not an exact-timing
  guarantee for finite-aperture waveforms). (c) **Round trip
  has no exact identity on one-sided finite apertures** (the mirrored −p
  background of the exact bi-infinite pair cannot cancel; a pure-ω test
  pinned this): the one-sided forward recovers the input *eventwise* —
  times exact, corr ≈ 0.96, scale ≈ 0.90–0.96 — documented as finite-
  aperture corrections rather than "interpolation error". (d) **Zero-offset
  trace preserved at x = 0 at the event level**: times within a sample,
  variant/train structure intact; amplitudes legitimately differ from M2 by
  the per-event focusing factor (e.g. T3-full's 1.8 s window pick shifts
  under neighbour interference; the primaries-variant picks are exact).
  (e) Independent forward anchor: a hand-built constant-V hyperbola gather
  transforms to `τ(p) = T₀√(1−p²V²)` picks. Anchors tested (32 new tests,
  179 total): T1 `2h·tanθ` closed form + hyperbola picks (θ = 0–40°) +
  stationary-phase amplitudes; T2 Snell refraction, 4-leg water-bottom
  multiple, multiple-vs-primary hyperbola separation (−12 ms @ 322 m,
  −35 ms @ 560 m — the corrected steeper-than-primary expectation);
  zero-offset events all models/variants; round trip; mute equivalence;
  validation errors. Also fixed an arithmetic slip in the corrected
  expectation text: V_rms(0.8 s) = 2001 m/s (was quoted 1790). All gates
  green (ruff, ty, pytest). Figures verified via glm-5.3-flash delegation:
  T1 (P1 + alternating train in `full` only, hyperbola 0.5→0.65 s,
  overlay on crest, no pre-0.4 s energy), T2 (P1/P2/M_wb1/M_wb2/peg12 all
  on their curves; M_wb1 curves down *much* faster than P2, P2 far-offset
  time 1.2–1.25 s vs theory 1.244 s), T3 (four primaries clean; M_s1 on
  the 1.0 s train; interbed23 below display clip by amplitude design,
  ~1 % of P1).
- 2026-11-09: **multiple-velocity expectation corrected (M4/M5 acceptance text)** —
  free-surface multiples repeat their generating primary's path, so their
  V_stk equals the RMS velocity of that (shallow) path and never exceeds the
  same-T₀ primary trend in a normally graded section (T2 water-bottom
  multiple: V_stk = v_w = 1500 m/s exactly vs V_rms(0.8 s) ≈ 2001 m/s →
  *steeper*; the radon-separability signature, a ridge pinned at the water
  velocity on semblance panels). The "flatter than primary / higher apparent
  velocity" phenomenon is an interbed effect requiring a reverberating layer
  faster than the deeper section (v_layer² > marginal RMS² of going deeper);
  none of T1–T3 has such geometry (T3's interbed generator is the soft layer,
  2300 over 3000 m/s) — a fast-layer-over-slow T4 is a candidate addition
  for M6 if that regime should be demonstrated.
- 2026-11-08: **M3 complete** — `reflectivity.stack_response`: the full ω–p
  plane-wave engine (same compound recursion and variant switches as M2,
  generalized to horizontal slowness): p-dependent interface coefficients via
  `_stack_terms`, layer round-trip *intercept* delays `2 q_i h_i`
  (`q_i = sqrt(1/v_i² − p²)`), free surface exactly −1 at every p.  Events in
  the angle gathers land on intercept times `τ(p) = Σ q_j h_j per leg`
  (decreasing with angle; `dτ/dp = −x` feeds M4).  `gathers.py`:
  `slowness_from_angle`, `angle_impulse_gather`, `angle_gather` (Ricker
  synthetic), `critical_mute_slowness`, `primary_intercept_times` (NaN where
  a leg is evanescent).  `viz.py`: `plot_angle_gather` (variable-density +
  dashed primary-intercept overlay + mute boundary) and
  `angle_gather_stack_figure`.  `scripts/angle_gathers.py` →
  `output/angle_gather_{model}_{variant}.png` + `_stack.png` for T1–T3.
  Deviations/design decisions: (a) **Engine convention is the conjugate of
  the `e^{−iωt}` physics**: traces are synthesized with `irfft` (delay =
  `e^{−iωt₀}`), which reproduces the physical solution only if coefficients
  are `conj` of model.py's standard ones — identical subcritical, conjugated
  post-critical (pinned by the T2 closed-form test at θ > 30°); the evanescent
  cosine branch `Im(c) < 0` then makes propagators decay `e^{−ω|q|h}`
  (stable tunnelling) instead of overflowing.  (b) **"Evanescent cutoff
  p > 1/v_max zeroed" implemented as a gather-level mute at
  `p = 1/max(vp[1:])` = smallest interface-critical slowness**: beyond it the
  stack is a lossless total reflector (`|U₀| = 1` exactly for `no_surface`) and
  the `full` free-surface series resonates (plane-wave cavity; measured
  `min|1+U₀| ≈ 5e-4` → `|S| ≈ 2e3` at 40°) — for T2/T3 the mute lands at 30°,
  for T1 at 41.8° (outside the 0–40° grid).  The engine itself stays exact,
  finite and warning-free across the whole grid (errstate test), including
  post-critical interface coefficients and tunnelling through evanescent
  fast layers.  (c) **Amplitudes are plane-wave pressures**: no geometric
  spreading, no obliquity factor — stated on every angle-gather figure
  (M3 risk item resolved).  (d) Closed-form anchors avoid angles exactly at
  criticality (e.g. θ = 30° for T2): there `R'(p) → ∞` amplifies ulp-level
  slowness noise to ~1e-8, so 29° is used instead.  Anchors tested (65 new
  tests, 147 total): `stack_response(p=0) == stack_response_normal` (≤1e-12,
  all models/variants); T1 exact closed forms at angle (`primaries`
  `R(p)e^{−2iωτ}`, `full` series `x/(1+x)`, machine precision); T2 exact
  two-interface closed forms (Snell-refracted times, coefficient products,
  incl. the conjugated post-critical branch); `full = no_surface/(1+` `no_surface)`;
  `|no_surface| ≤ 1` (lossless energy bound, = 1 past critical); Hermitian
  response; θ=0 gather column == M2 trace (≤2e-14); T1 envelope picks
  (time `0.5·cosθ`, amplitude `R₁(θ)`); intercept-time values and NaN cutoffs;
  input validation.  Figures verified via glm-5.3-flash delegation: T1 stack
  (primary 0.5 s + 1.0 s surface multiple full-only, moveup toward earlier
  times, no pre-event energy), T2 stack (primaries 0.4/0.9 s, 0.4 s-period
  water-bottom train full-only, AVO build-up to the 30° mute), T3 stack
  (primaries 0.5/1.0/1.3/1.8 s, surface train full-only, interbed energy
  no_surface-only, P4 curve correctly ending at 36.9°).
- 2026-11-06: **M2 complete** — `wavelet.py` (unit-peak Ricker time wavelet +
  analytic spectrum `W(f) = 2 f²/(√π f_dom³)·exp(−(f/f_dom)²)` with
  `rfft(w) = W/dt` consistency, tested against a fine FFT of the sampled
  wavelet), `reflectivity.py` normal-incidence path (`stack_response_normal`
  compound recursion per frequency: `G_i = R_i + Td_i·Tu_i·X/(1 + R_i·X)`,
  `X = G_{i+1}·e^{−2iωτ}`; surface series `U₀/(1 − R_s·U₀)` with
  `FREE_SURFACE_REFLECTIVITY = −1`; `variant_multiples` switch table shared by
  the M3 engine-to-be; `impulse_response`/`zero_offset_trace` with an `n_fft`
  extension parameter), `viz.py` M2 helpers, `scripts/zero_offset.py` →
  `output/zero_offset_{model}_{variant}.png` plus a `{model}_stack.png`
  comparison per model. Deviations/design decisions: (a) **T1 anchor times
  corrected** — the original acceptance said arrivals at (n+1)·1.0 s, but with
  h = 500 m, vp = 2000 m/s the round trip is 2h/v = 0.5 s; data, the PLAN table
  (TWT 0.5 s) and AGENTS (“period 2h/v”) were already consistent, the 1.0 s was
  a planning slip. (b) **Pure water-bottom train times end at 2.0 s**: from
  2.4 s on (`0.4·6 = 0.4·1 + 0.5·4`) sediment-reverberation events coincide
  with the train, so the simple series `R₁(−R₁)ⁿ` no longer holds there.
  (c) **FFT wraparound needs long records**: the impulse response tail beyond
  the FFT record folds back (a dense 0.1 s event comb of peg-leg/interbed
  events persists past the visible window; measured fold-back: ~2e-5 at
  T=12 s → 6e-16 at T=48 s for T2), so `impulse_response`/`zero_offset_trace`
  take `n_fft`; the 1e-6-level anchor tests use T_fft = 8–32 s and figures use
  8× the display window. (d) Synthetic spectrum normalization is
  `S·W/dt` (not `W·dt`): pinned by the frequency-vs-time-convolution test.
  (e) `LayeredModel.transmissions_normal` added (pressure T pair,
  `Td·Tu = 1 − R²`, single-sourced for the recursion). All gates green
  (ruff, ty, 82 tests incl. new test_wavelet/test_zero_offset). Figures
  verified via glm-5.3-flash delegation: T2 stack (2 primaries, alternating
  train in `full` only, no pre-0.35 s energy) and T3 stack (4 primaries,
  0.5 s surface train `full`-only, interbed energy `no_surface`-only).
- 2026-11-06: **M1 complete** — `units.py` (single degree/radian + s/ms conversion
  point), `model.py` (`LayeredModel` dataclass; normal-incidence and p-dependent
  acoustic R/T; flux-normalized `T_hat` with `|R|^2+|T_hat|^2 = 1`; complex-safe
  post-critical behaviour `|R| = 1, T_hat = 0`; critical angles), `io.py`
  (3-column loader; two 2-column files merged on depth with tolerance;
  blocked-model loader; uniform-TWT blocking with exact slowness integration,
  harmonic-mean layer velocities, exact piecewise-linear density averaging).
  Canonical models T1–T3 in `data/`. Deviations/design decisions: (a) canonical
  `data/*.csv` are *blocked* logs — `depth_m` is the layer top, last row =
  half-space, read by `load_blocked_model` — so layer boundaries stay exact
  (a sampled-log read would smear them); (b) energy-normalized transmission is
  flux-based (vertical energy flux), hence exactly 0 in the evanescent region;
  tests cover the full 0–40° grid including past the T2 water-bottom critical
  angle (38.68°); (c) blocking covers the whole logged interval with uniform-TWT
  cells (deepest cell absorbs the remainder, so no layer is thinner than the
  target TWT) and appends a half-space duplicating the deepest cell — base
  interface R = 0 exactly, total TWT/mass preserved. All gates green (ruff,
  ty, 50 tests). Demo: ``scripts/model_demo.py`` (verified via glm-5.3-flash
  delegation) produces ``output/model_demo_{models,rtheta,blocking}.png``.
- 2026-11-06: **M0 complete** — uv-managed scaffold (Python 3.14.9, src/mm1d
  layout, hatchling build), ruff (lint+format, notebooks excluded so the
  prototype `1dmm_ud.ipynb` stays untouched), ty, pytest configured; stub modules
  with docstrings per AGENTS.md layout; `WavefieldVariant` literal defined in
  `reflectivity.py`; 3 smoke tests; all gates green; `uv.lock` committed.
- 2026-11-06: untracked root reference PDFs (Müller, CRR, Yilmaz, …) — they stay
  on disk as local-only material and are gitignored (`/*.pdf`); M0 commit
  amended so the ~92 MB of blobs never entered history.
- 2026-02-15: resolved free-surface convention to R_s = −1 (pressure-release);
  T1/T2 analytic anchors updated to alternating-polarity multiple trains;
  verified against exact packet recursion.
- 2026-02-15: initial plan (M0–M7), acoustic-only + Ricker scope, canonical
  models T1–T3 defined.
