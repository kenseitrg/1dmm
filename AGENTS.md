# AGENTS.md — mm1d: 1.5D Reflectivity Synthetic Modelling Tool

## Project Overview

`mm1d` ("1D multiple modelling") is a Python library for computing synthetic seismic
wavefields from a pair of well logs (P-velocity and density), using the reflectivity
method for a horizontally layered (1.5D) earth. Its purpose is **multiple analysis**:
generating and separating wavefield components so that multiple-generating horizons
and the traveltimes of major multiple energy contributions can be identified.

The user-facing goal: feed in `vp`/`rho` logs, get out

1. **Zero-offset (normal-incidence) synthetics** for three wavefield types (below).
2. **1.5D angle gathers** for the same three wavefield types.
3. **Offset gathers** obtained by angle→offset conversion (Snell's-law ray tracing).
4. **Semblance panels** computed for each wavefield type.
5. **Theoretical RMS (stacking) velocity curves** for labelled events, overlaid on the
   semblance panels to aid event identification.

### The three wavefield types

Every modelling output is produced in three variants, differing by which multiple
mechanisms are enabled:

| Variant | Primaries | Surface-related multiples | Interbed multiples |
|---|---|---|---|
| `full`            | yes | yes | yes |
| `no_surface`      | yes | no  | yes |
| `primaries`       | yes | no  | no  |

These correspond to Müller's "partial responses": the full medium response, the
original reflectivity-method response (source radiation downward only, no free-surface
reflections back into the stack), and the single-scattering (Born/primary) response.
In the reflectivity recursion they are selected by boundary/coefficient switches:
free-surface reflector on/off for `full`/`no_surface`, and suppression of reverberation
within the stack for `primaries`. This is exactly the decomposition prototyped in the
existing notebook `1dmm_ud.ipynb` — **but note**: the notebook's free-surface value
`reflectivity_down[0] = +1` was a toy simplification; the physically consistent value
for our coefficient convention is `−1` (see Sign conventions below).

## Reference Material (in repository root — do not delete)

- `330-Article-583-1-10-20230415.pdf` — G. Müller (1985), *The reflectivity method:
  a tutorial*, J. Geophys. 58:153–174. **The** methodological reference: plane-wave
  reflection/transmission coefficients, recursive (numerically stable) computation of
  reflectivities and transmissivities in the frequency–slowness (ω–p) domain, and
  extraction of partial responses (multiples on/off).
- `CRR201673.pdf` — Todorov & Margrave (2016), *RefMod: software program for
  reflectivity modeling*, CREWES Report 28. A practical implementation walkthrough of
  the same method (Java): layer-parameter setup, ω–p algorithm steps, toggling
  multiples on/off.
- `1dmm_ud.ipynb` — the project prototype: normal-incidence ray tracing producing the
  three wavefield variants. Use it as a cross-check for the library implementation.
- Root reference PDFs (all of the below) are **local-only**: untracked and
  gitignored (`/*.pdf`) — keep them on disk, do not delete, do not re-add.
- Remaining PDFs (`geokniga-seismicdataanalysis.pdf`, `s10518-017-0206-7.pdf`,
  `artefakty-...pdf`) are background reading; the first is Yilmaz, *Seismic Data
  Analysis* (semblance, NMO/stacking velocity, multiple terminology).

## Method Summary (what the code must implement)

1. **Model building** — load `depth, vp, rho` logs; resample/block onto a uniform
   two-way-time grid or a discrete layer stack; compute normal-incidence and
   angle-dependent reflection/transmission coefficients (acoustic; Zoeppritz-free:
   P-wave acoustic approximation with `vp`, `rho`).
2. **Reflectivity computation (ω–p domain)** — for each horizontal slowness `p`
   (equivalently incidence angle) and frequency `ω`, recursively compute the total
   reflectivity/transmissivity of the layer stack. Use Müller's recursive
   compound-matrix formulation, *not* propagator matrices (stability at all `p`).
   Evanescent region `p > 1/v_max` must be excluded (zeroed).
3. **Wavefield assembly** — combine source wavelet spectrum, free-surface receiver
   response and the stack reflectivity; inverse transform (FFT over ω) to time.
   The three variants are selected as described above.
4. **Gathers** — zero-offset trace (θ = 0) and angle gathers (θ = 0…θ_max); conversion
   to offset gathers via Snell's-law ray tracing through the layer stack:
   `x(θ) = Σ hᵢ tan θᵢ`, `T(θ) = Σ hᵢ/(vᵢ cos θᵢ)` per leg (legs depend on event path).
5. **Semblance** — Neidell–Taner semblance over (T₀, V_stk) on the offset gathers,
   one panel per wavefield type.
6. **Event catalog & theoretical RMS velocities** — enumerate event ray paths
   (primaries per interface; surface multiples of order n; interbed multiples between
   interface pairs; peg-legs). For each event compute total zero-offset traveltime and
   effective stacking velocity from its path legs,
   `V_stk = sqrt(Σ t_leg v_leg² / Σ t_leg)`, and overlay the (T₀, V_stk) trajectory on
   the corresponding semblance panel. This is the multiple-identification payoff of
   the tool.

Keep this pipeline staged so each step is independently runnable and testable:
`logs → model → reflectivity → synthetics → gathers → semblance → events/viz`.

The implementation roadmap with milestones, acceptance criteria, and test-model
definitions lives in `PLAN.md`. Keep it updated: mark milestones complete as they
finish, and record deviations from the plan there.

### V1 simplifications (explicit)

- **Acoustic P-waves only** — `vp` + `rho`; no shear, no converted waves, no mode
  conversions.
- **Ricker wavelet** as the source for all v1 wavefield generation.
- These are the first expansion candidates afterwards (elasticity, arbitrary
  wavelets, Q).

## Tech Stack & Tooling

- **Python 3.15**, managed with **uv** (never pip/venv directly).
- Dependencies: `numpy`, `scipy`, `matplotlib`. Add more only when genuinely needed
  (no pandas for CSV — `numpy.loadtxt`/`genfromtxt` is enough).
- **ruff** for linting and formatting; **ty** for type checking. Both run via uv
  (`uv run ruff …`, `uv run ty …`) — do not install globally.
- **pytest** for tests.

### Commands

```bash
uv sync                      # create/update .venv from pyproject.toml + uv.lock
uv run python scripts/…      # run an analysis script
uv run pytest                # run the test suite
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type-check
```

Before declaring any task complete, `ruff check`, `ty check` and `pytest` must all
pass.

## Project Layout

```
1dmm_v2/
├── AGENTS.md                 # this file
├── pyproject.toml            # package config, ruff + ty + pytest config
├── uv.lock
├── src/
│   └── mm1d/                 # the library
│       ├── __init__.py
│       ├── io.py             # CSV/TXT log loading, resampling, blocking
│       ├── model.py          # layered model, reflectivity coefficients
│       ├── reflectivity.py   # ω–p recursion, wavefield variants (full/no_surface/primaries)
│       ├── wavelet.py        # Ricker wavelet (v1 source) and its spectrum
│       ├── gathers.py        # zero-offset, angle gathers, angle→offset conversion
│       ├── semblance.py      # semblance panels
│       ├── events.py         # event path catalog, traveltimes, theoretical RMS velocities
│       ├── viz.py            # plotting: synthetics, gathers, semblance with overlays
│       └── units.py          # unit constants/conversions (single source of truth)
├── scripts/                  # runnable end-to-end workflows producing figures
├── data/                     # example input logs (CSV)
├── tests/                    # pytest suite, incl. analytic-model validation
├── output/                   # generated figures (gitignored)
└── *.pdf, 1dmm_ud.ipynb      # reference material (root, untouched)
```

## Conventions

### Units (SI everywhere internally)

- depth `m`, layer thickness `m`, time `s`, frequency `Hz`, velocity `m/s`,
  density `kg/m³`, slowness `s/m`.
- Angles: **degrees at the public API**, radians internally (convert at the boundary,
  in one place). Same for ms/s if a UI layer ever appears: convert at the edge only.

### Sign & amplitude conventions (single-sourced, tested)

- Interface reflection (down-going → up-going):
  `R = (Z_below − Z_above)/(Z_below + Z_above)` — pressure convention, SEG-normal
  polarity (impedance increase → positive event). Round-trip transmission
  `T_down·T_up = 1 − R²`.
- **Free surface: `R_s = −1`** (pressure-release surface). The physically equivalent
  geophone/velocity picture (`R_s = +1`) flips every interface coefficient's sign too;
  mixing them (notebook's +1 with standard interface R) is non-physical. With
  `R_s = −1` the surface-multiple train alternates polarity: `A_n = R·(−R)ⁿ` —
  verified against the exact packet recursion.
- Recorded trace = up-going wavefield just below the surface (de-ghosted
  hydrophone-equivalent); primaries are unaffected by `R_s`, only the surface-multiple
  train's polarity pattern depends on it.
- At oblique incidence (ω–p engine) the acoustic free-surface coefficient stays
  exactly `−1` for all subcritical angles (angle-dependent surface response only
  appears with elasticity — out of scope).

### Input data format

CSV or whitespace-separated text. Either a single file with columns
`depth_m, vp_m_s, rho_kg_m3`, or two two-column files (`depth_m, vp_m_s` and
`depth_m, rho_kg_m3`) merged on depth (with tolerance). Loaders validate for
monotonic depth, positivity, and NaNs; fail loudly with a clear message.

### Code style

- Numpy-first: functions are pure, operate on plain ndarrays (`NDArray[np.float64]`),
  return new arrays; no mutable global state, no classes where functions suffice
  (a light `LayeredModel` dataclass is fine).
- Every public function fully type-annotated (ty must pass); NumPy-style docstrings
  with units stated for every physical quantity.
- Prefer small, individually testable functions over monolithic pipelines; scripts
  are thin orchestrations of library calls.
- Figures: every script writes figures to `output/` with deterministic filenames and
  also shows them (guard with a `--no-show` flag).
- The three wavefield variants are selected by a shared enum/literal
  `{"full", "no_surface", "primaries"}` — one name everywhere (docs, code, plots).

## Testing / Validation Requirements

Physics correctness is checked against analytic results — numerical regression
baselines are not enough. Test models use **thick layers** (layer two-way time ≥
~3 dominant wavelet periods, e.g. ≥ 250 ms TWT for a 20 Hz Ricker) so that events
are separated and **interpretable by eye**; canonical models live in `data/`
(see `PLAN.md` for their definitions):

- **Single interface (two half-spaces)**: first arrival amplitude equals the
  plane-wave reflection coefficient; full-wavefield multiple train has exact
  geometric-series amplitudes and period `2h/v` (water-bottom multiple test).
- **Energy conservation**: `R² + T² = 1` at normal incidence, lossless case.
- **Variant consistency**: `full − no_surface` contains only surface-multiple energy;
  `no_surface − primaries` only interbed energy; zero above the first return.
- **Primaries**: event times equal interface two-way times; small-spread stacking
  velocities satisfy Dix from layer velocities.
- **Angle gathers**: at θ = 0 the angle gather matches the zero-offset trace;
  angle→offset conversion round-trips against direct ray tracing.
- **Theoretical RMS curves**: for primary events, curves pass through the semblance
  maxima of the `primaries` panel.

## Visual Figure Verification (multimodal delegation)

Not every model running this project can see images. **If the current model
cannot view images**, verify generated figures (event separation, overlay
correctness, plotting sanity) by delegating to a standalone Pi-agent session with
a multimodal model:

```bash
pi --provider zai --model glm-5.3-flash --print \
    "<precise question about the figure> @output/<figure>.png"
```

- `glm-5.3-flash` is confirmed to accept image input; `--provider zai` resolves
  the model-ID ambiguity across providers (verified working with this exact form).
- Ask targeted questions ("How many reflection events are visible before 1 s?",
  "Does the dashed curve pass through the semblance maximum?") — not "is this OK?".
- The answer arrives on stdout; treat it as evidence in the usual way.

## Out of Scope (for now)

- Elastic/full-waveform effects: shear logs, mode conversions, Zoeppritz.
- Attenuation (Q), anisotropy, source/receiver directivity beyond plane waves.
- 2D/3D geology, lateral heterogeneity.
- LAS loading (CSV/TXT only), GUIs, SEGY export.
