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

### M4 — Angle → offset conversion (`gathers.py`)
**Goal**: offset gathers via Snell-law ray tracing.
- Per-layer ray parameter: p = sinθ_i/v_i constant; x = Σ hᵢ tanθᵢ, T = Σ
  hᵢ/(vᵢ cosθᵢ) over the event's legs; event-agnostic gather transform
  (map θ→x per trace, resample amplitude to uniform offset grid).
- Script `scripts/offset_gathers.py` → `output/offset_gather_{model}_{variant}.png`.
**Done when**:
- Zero-offset trace preserved at x = 0; offset of T1 primary vs θ matches
  closed-form 2h·tanθ; round-trip angle↔offset consistent within interpolation
  error; near/far offsets of the T2 water-bottom multiple follow the theoretical
  moveout (flatter than primary — the classic pitfall this project exploits).

### M5 — Semblance (`semblance.py`)
**Goal**: Neidell–Taner semblance panels per variant.
- Sliding-window semblance over (T₀, V_stk) on hyperbolic moveout; shared for
  all three variants; parameter ranges auto-scaled from gather extent.
- Script `scripts/semblance.py` → `output/semblance_{model}_{variant}.png`.
**Done when**: on T2 `primaries` panel, maxima at (0.4 s, V_rms₁) and
(0.9 s, V_rms₂) within one grid cell of Dix values; on `full` panel the
water-bottom multiple ridge appears at *higher* apparent velocity than a primary
at the same T₀ would (moveout of 2·t_water-bounce legs).

### M6 — Event catalog & theoretical velocities (`events.py`, `viz.py` overlays)
**Goal**: the multiple-identification payoff.
- Path enumeration: primaries per interface; surface multiples (order n, any
  reflector); interbed multiples (reflector pair, order n); peg-legs (surface +
  one interbed leg). Bounded order (default ≤ 3) and amplitude threshold (default
  |A| ≥ 0.01 of primary at same interface) to keep catalogs human-scale.
- Per event: zero-offset T₀ (sum of legs), amplitude product (R/T chain),
  stacking velocity V_stk = sqrt(Σ t_leg v_leg² / Σ t_leg).
- Overlay plotting: (T₀, V_stk(T₀-offset)) trajectories on semblance panels,
  event table printout.
- Script `scripts/event_catalog.py` on T2 + T3 →
  `output/semblance_events_{model}_{variant}.png` + `output/events_{model}.md`.
**Done when**: primary curves pass through `primaries` semblance maxima (visual +
numeric spot check); each cataloged surface multiple in T2 lands on a visible
`full`-panel ridge; catalog explains every ridge visible in T2/T3 panels (no
unexplained strong events).

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
- **Semblance bias for interfering events** in `full` gathers: compare against
  variant-separated panels — this is a feature of the tool, not a bug; make sure
  M6 report phrasing reflects it.

## Changelog

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
