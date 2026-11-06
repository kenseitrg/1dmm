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
AGENTS.md sign conventions): arrivals at tₙ = (n+1)·1.0 s with amplitudes
R₁·(R_s·R₁)ⁿ = R₁·(−R₁)ⁿ — alternating polarity: +0.304, −0.093, +0.028, −0.009
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

### M1 — Model building (`io.py`, `model.py`, `units.py`)
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

### M2 — Zero-offset synthetics, three variants (`wavelet.py`, `reflectivity.py` normal-incidence path)
**Goal**: `1dmm_ud.ipynb` logic, productized.
- Ricker wavelet (time + frequency domain, consistent normalization).
- Normal-incidence recursion for the three variants (switch semantics per
  AGENTS.md table); free-surface convention **R_s = −1** (pressure-release),
  single-sourced in code and validated by the alternating-polarity multiple train
  (resolved at planning time — notebook's `+1` was a toy simplification).
- Script `scripts/zero_offset.py`: three stacked traces for T2 and T3 →
  `output/zero_offset_{model}_{variant}.png`.
**Done when**:
- T1: `full` peaks at (n+1)·1.0 s match R₁·(R_s·R₁)ⁿ within 1e-6 (impulse
  response, no wavelet); `primaries` single arrival R₁.
- T2: water-bottom multiple amplitudes follow the analytic series including
  transmission terms; `full − no_surface` and `no_surface − primaries` vanish
  above 0.35 s (before first return).
- Figure check (M2 figure): events countable by eye, variants visibly differ.

### M3 — ω–p reflectivity engine → angle gathers (`reflectivity.py` full)
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
