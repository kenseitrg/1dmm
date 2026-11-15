# Handover: clean offset gathers and numerical fixes

## 1. Read this first

The user requested a review of the project plan/implementation, especially removal
of slant-stacking artifacts in offset gathers. After the review, the user selected
**clean diagnostic gathers**, rather than a physical full-wavefield backend or
only improving the current slant stack. They then requested these handover notes
so a smaller model can implement the work.

**Agreed direction:** add ray-path diagnostic synthesis alongside the existing
reflectivity/slant-stack backend. Retain the independently implemented omega-p
recursion for validation. Bring path enumeration forward from M6 before further
semblance tuning. Fix the confirmed numerical defects in the old backend too.

This document is an implementation guide, not a claim that any fixes are done.
The review and handover changed no implementation files. API names, diagnostic
amplitude normalization, and numerical acceptance budgets below are **proposals**;
do not describe them as already approved or implemented.

### Existing work: preserve it

At handover, `git status --short` showed:

```text
 M PLAN.md
 M src/mm1d/semblance.py
 M src/mm1d/viz.py
?? scripts/semblance.py
?? tests/test_semblance.py
```

These changes predate this handover. Do not reset, overwrite, or discard them.
`HANDOVER.md` is a new file added by this task. Inspect the current diff before
starting: the tree may have changed since these notes were written.

Baseline: `uv run ruff check .`, `uv run ty check`, and `uv run pytest -q` all
passed, with **199 tests**. Passing existing tests does not cover the defects below.

Read `AGENTS.md`, relevant sections of `PLAN.md`, and the files named below.
Use `uv`, never pip. Do not delete or add the local-only reference PDFs to git.
AGENTS requests Python 3.15, while pyproject/ty currently target 3.14; do not
bundle a runtime migration with these fixes. Use the existing uv environment and
record the documentation/configuration discrepancy for later reconciliation.

## 2. Scope and non-goals

### Deliver

1. Correct quadrature and response-aware sampling in existing offset synthesis.
2. Bounded, tested event-path enumeration and path coefficient calculations.
3. Clean diagnostic gathers with Snell-law moveout and unchanged Ricker wavelets.
4. Independent comparisons against analytic models and omega-p reflectivity.
5. Semblance/plot integration with explicit backend and amplitude labels.
6. Updated roadmap, documentation, and reproducible artifact checks.

### Do not silently change

- The existing `offset_gather` amplitude convention or default backend.
- The pressure sign convention: interface impedance increase gives positive R;
  `FREE_SURFACE_REFLECTIVITY = -1`.
- SI units, degrees at public angle APIs, or the three shared variant names.
- The v1 exclusion of post-critical/evanescent contributions.
- The meaning of the current zero-offset plane-wave synthetic.

Do not claim the new diagnostic gathers are full physical point/line-source
Green functions. They deliberately omit diffraction and other non-ray effects.
Do not claim a finite catalog contains every multiple without convergence evidence.
Do not conceal artifacts with stronger display clipping, arbitrary mutes, added
noise, or a forced replacement of the x=0 trace.

## 3. Code map

| File | Relevant code |
|---|---|
| `src/mm1d/gathers.py` | `_aperture_weights`, `_refine_factor`, `_refine_angles`, `_slant_stack_spectra`, `offset_gather`, `offset_impulse_gather` |
| `src/mm1d/gathers.py` | `SURFACE`, `path_offset_time`, `primary_offset_times`, `critical_mute_slowness` |
| `src/mm1d/reflectivity.py` | `variant_multiples`, `FREE_SURFACE_REFLECTIVITY`, `stack_response`, `stack_response_normal` |
| `src/mm1d/model.py` | `LayeredModel`, pressure reflection/transmission APIs |
| `src/mm1d/wavelet.py` | `ricker_wavelet`, `ricker_spectrum` |
| `src/mm1d/events.py` | Currently only a docstring: M6 is not implemented |
| `src/mm1d/semblance.py` | `semblance_panel`, `scan_slowness_bound`, `dix_rms_velocities` |
| `scripts/offset_gathers.py` | Current demonstrations, hard-coded example paths, overlays |
| `scripts/semblance.py` | Scan aperture, seeded 1% noise floor, local-pick checks |
| `tests/test_offset_gathers.py` | Existing analytic anchors and artifact tests |
| `tests/test_semblance.py` | Existing semblance tests; currently untracked user work |

Current offset synthesis integrates `U(p, omega) * cos(omega*p*x)` with an
`|omega|` filter, taper, and finite p aperture. Its wavelet/amplitudes differ from
plain Ricker path deposition. A samplewise comparison between those two offset
backends is therefore **not** a valid equality test.

## 4. P0-A: fix aperture quadrature

### Confirmed defect

In `_aperture_weights`, when a critical mute exists, `right[-1] = p_cut` even
when the requested angle grid ends well below the cut. With `taper_fraction=0`,
the last weight integrates beyond the requested aperture.

Reproduce from repository root:

```bash
uv run python - <<'PY'
import numpy as np
from mm1d.io import load_blocked_model
from mm1d.gathers import _aperture_weights

m = load_blocked_model('data/water_bottom.csv')
for top in (10., 20., 40.):
    p, w = _aperture_weights(m, np.arange(top + 1), True, 0.)
    end = min(p[-1], 1 / max(m.vp_m_s))
    print(top, 'weight sum:', w.sum(), 'nominal interval:', end - p[0],
          'relative error:', w.sum() / (end - p[0]) - 1)
PY
```

Before fix: approximately **+93.97%** for 10 degrees and **+23.10%** for
20 degrees. The 40-degree case also exposes an endpoint-coverage discrepancy
(about -1.52% against the nominal cut), not the same over-extension bug.
Default tapering suppresses the last-node weight for short apertures; it does
not make the untapered API correct.

### Implementation checklist

- [ ] First add a failing constant-integrand test for apertures entirely below
  the mute: weights must sum to `p[-1] - p[0]` when tapering is off.
- [ ] Ensure the requested endpoint, not a more distant mute, ends quadrature.
- [ ] Define the cutoff-straddling interval policy explicitly. Either evaluate
  a safe subcritical endpoint or stop at the last evaluated valid node and
  document/test the omitted interval and convergence. Never pretend that a
  simple endpoint-weight adjustment exactly integrates an unevaluated segment.
- [ ] Never evaluate at the singular critical point. Account for near-critical
  resonances when selecting/testing any inserted endpoint.
- [ ] Test nonuniform grids, mute disabled, aperture below/across the mute,
  no critical mute, taper on/off, nonnegative weights, and muted-node exclusion.
- [ ] Review existing `test_aperture_weights_mute_exclusion`: it asserts the old
  weight formula. Replace that expectation with the chosen mathematical policy,
  not another implementation-specific formula.

Keep changes targeted. This is an independently shippable correctness fix.

## 5. P0-B: make slowness sampling response-aware

### Confirmed defect

`_refine_factor` uses only `delta_p <= 1/(4*f_max*x_max)` and immediately returns
1 for x_max=0. This resolves the cosine kernel but not the oscillatory response
`U(p, omega)`. The same x=0 trace changes substantially depending on which other
offsets are requested in the call.

Reproduce:

```bash
uv run python - <<'PY'
import numpy as np
from mm1d.io import load_blocked_model
from mm1d.gathers import offset_gather

m = load_blocked_model('data/thick_5layer.csv')
a = np.arange(41.)
fine = np.arange(0., 40.0001, .0625)
dt, n = .002, 2000
ref = offset_gather(m, 'primaries', fine, np.array([0.]), dt, n, 20.,
                    n_fft=8*n)[0]
for offsets in ([0.], [0., 1000.], [0., 4500.]):
    got = offset_gather(m, 'primaries', a, np.array(offsets), dt, n, 20.,
                        n_fft=8*n)[0]
    print(offsets, np.max(np.abs(got-ref)) / np.max(np.abs(ref)))
PY
```

Before fix, relative maximum differences are approximately:

```text
[0]          0.0976063
[0, 1000]    0.00003523
[0, 4500]    0.000000215
```

### Implementation guidance

For an event, `U ~ A(p) exp(-i*omega*tau(p))`, with
`d tau / dp = -x_event(p)`. Thus the phase derivative of the synthesis
integrand includes the event path as well as the requested receiver offset.
The cosine kernel alone is not a sufficient sampling criterion. Multiple
paths can oscillate more rapidly than primaries.

- [ ] Add failing single-offset versus batched-offset and fine-grid tests first.
- [ ] Retain kernel sampling as a lower bound, but add response/output convergence
  checks under p-grid refinement, or a justified conservative response-aware
  bound backed by those checks. Do not just increase the constant or the cap.
- [ ] Test convergence across multiple refinements; one accidental agreement is
  not enough. Compare waveforms, not just event peak times.
- [ ] Cover T1/T2/T3, all variants, x=0, near/far offsets, and impulse/Ricker
  bandwidths. Keep expensive parameter combinations appropriately limited.
- [ ] Use both absolute and relative tolerances: near-zero traces need an
  absolute floor. Proposed initial Ricker target: errors below `1e-3` of a
  clearly defined reference-trace/gather peak on canonical models, then tighten
  where practical. This is a proposed engineering budget, not a measured result.
- [ ] A resource/refinement cap must warn or fail if tolerance is unmet; do not
  silently label the result alias-free. Expose useful convergence diagnostics.
- [ ] Preserve offset order and negative-offset symmetry. Shared traces in
  separate calls must agree within the documented tolerance, not necessarily
  bit-for-bit if adaptive grids differ.
- [ ] Benchmark runtime/memory and avoid storing unnecessarily large
  p-by-frequency-by-offset arrays.

Fixing this error does **not** eliminate finite-aperture smearing. Do not promise
artifact-free gathers from refinement alone.

## 6. P1-A: implement bounded event paths (bring forward M6)

Work in `events.py`, with focused tests in a new `tests/test_events.py`.
Suggested representation: a small immutable event record with turning points,
layer traversal counts, reflection/transmission interactions, family metadata,
and zero-offset properties. Keep functions pure and typed.

### Reuse and conventions

`SURFACE = -1`; interface k is the base of finite layer k. Examples:

```text
Primary k:                 [-1, k, -1]
Surface multiple, order n:  [-1, k] * (n+1) + [-1]
Interbed, a < b:            [-1, b, a, b, -1]
Mixed surface path:         [-1, 1, -1, 0, -1]
```

`path_offset_time` already computes geometry for turning-point sequences.
Its permissive geometry API is not a validator of physically alternating
reflection directions; enumeration must produce valid paths.

A useful implementation is a directional ray-state traversal:

1. Start down-going immediately below the surface with amplitude 1.
2. Propagate through a layer to the next interface; branch into reflection
   and transmission, recording the interaction and traversed layer.
3. A downward reflection has coefficient `R_i(p)`; an upward reflection
   from the underside has `-R_i(p)` in the subcritical pressure convention.
   Transmissions use the appropriate `T_down` or `T_up`, not energy-normalized T.
4. At a surface return, record the up-going event **before** surface reflection.
   For `full`, optionally continue down with `R_s=-1`; otherwise terminate.
5. Transmission into the bottom half-space escapes and never returns.

Variant rules must use the shared names/semantics:

- `full`: all allowed interactions within the explicit bounds.
- `no_surface`: no continuation after a surface return; interbed reflections remain.
- `primaries`: only one downward-to-upward reflection, with transmission losses
  on both legs; no internal upward-to-downward reflections and no surface bounce.

Do not multiply by R_s at the receiver or impose R_s on the initial source.
Use coefficients at a shared horizontal slowness p. `model.reflection(i, angle)`
expects the **local** angle above interface i, not the surface incidence angle.
The public p-based coefficient functions avoid that ambiguity.

### Enumeration safety and completeness

- [ ] Bound by recording time plus the chosen wavelet-tail margin, explicitly
  defined reflection/order limits, and a maximum path count/resource guard.
- [ ] Define what “order” counts. Do not confuse surface order, internal order,
  and total reflections. Do not silently use the M6 display limit of order 3
  as a complete synthesis catalog.
- [ ] Separate a compact display catalog from the larger synthesis path set.
- [ ] Sum distinct paths coherently, including paths with identical moveout.
  Do not deduplicate by arrival time or layer counts and lose multiplicity.
  Aggregation is allowed only after summing their coefficient contributions.
- [ ] Do not prune a branch solely because its current pressure amplitude is
  small: individual transmission coefficients can exceed one. A zero-offset
  amplitude cutoff also does not bound oblique amplitudes. Treat any such
  truncation as heuristic unless a conservative bound is proved.
- [ ] Report which bounds were reached; demonstrate stability as order/path
  limits increase and amplitude thresholds decrease. Never silently truncate
  a path explosion on fine logs.

### Independent acceptance tests

- T1: `R * (-R)**n`, arrival times `(n+1)*2*h/v`; no internal multiples.
- Primary k: zero-offset amplitude
  `R_k * product_{i<k}(T_down_i*T_up_i)` and interface TWT.
- T2/T3: hand-built surface, internal, and mixed paths; underside signs;
  transmission losses; distinct orderings with coincident moveout.
- At selected subcritical p, independently sum
  `A_path(p)*exp(-i*omega*tau_path(p))` and compare with `stack_response`.
  Primaries should agree to numerical precision; multiple variants should
  converge as bounds increase. In a bounded time-window comparison, account
  for wavelet tails and FFT periodicity; a time-truncated series is not an
  exact frequency-domain match to the infinite response.
- Compare normal-incidence path synthetics to `zero_offset_trace` with enough
  FFT padding and converged path limits.

Do not reimplement the reflectivity recursion inside the catalog just to make
this comparison pass. Independent formulations are the point of the test.

## 7. P1-B: add clean diagnostic offset synthesis

Prefer an additive public function (suggested name `ray_offset_gather`) rather
than replacing `offset_gather`. Confirm the public API and amplitude convention
with the user before committing a consequential interface change.

**Recommended initial amplitude convention:** dimensionless pressure R/T path
products, unit-peak Ricker wavelets, no geometric spreading. This makes x=0
comparable to the existing plane-wave synthetic and prioritizes interpretation.
It is not the current slant-stack focusing convention and not a physical
point-source amplitude. Label it explicitly. This recommendation still needs
confirmation; the user selected clean gathers, not a specific amplitude law.

### Geometry and deposition

For path traversal counts `n_i`, finite-layer velocities `v_i`, thicknesses
`h_i`, and subcritical horizontal slowness p:

```text
q_i(p) = sqrt(1/v_i**2 - p**2)
x(p)   = sum_i n_i*h_i*p/q_i(p)
T(p)   = sum_i n_i*h_i/(v_i**2*q_i(p))
tau(p) = sum_i n_i*h_i*q_i(p) = T(p) - p*x(p)
T0     = sum_i n_i*h_i/v_i
Vrms   = sqrt(sum_i n_i*h_i*v_i / T0)
```

Sum only over traversed layers; do not let `0 * NaN` contaminate a valid path.
For x=0 use p=0 exactly. For x>0, x(p) is monotone for a valid propagating path;
solve `x(p)=abs(x_requested)` with a bracketed solver such as SciPy `brentq`.
Verify with `path_offset_time` and closed-form constant-velocity hyperbolas.

Keep the initial support within the existing global subcritical bound and any
explicit requested angle aperture. Do not relax the global cutoff merely
because a shallow path stays propagating: that changes current v1 policy.
If no admissible ray reaches an offset, report it as unsupported; do not clip
the ray parameter and place an event at the wrong time. Backend validity/support
must be visible to plotting/scanning rather than interpreted as measured zeros.
No slant-stack taper is needed for direct deposition, but finite ray support
still limits the available offsets.

For each supported event/offset, evaluate directly:

```text
gather[ix, :] += A_path(p) * ricker_wavelet(t_samples - T(p), f_dom)
```

This handles fractional arrival times without rounding impulses to samples and
without periodic FFT wrap-around. Optimize to a tolerance-controlled local
wavelet window later if needed. Account for events centered just beyond the
record whose wavelet tails enter it. Ricker is zero-phase, not strictly causal:
its legitimate leading lobe is not an artifact.

### Acceptance tests

- [ ] T1 primary and multiple hyperbolas; exact analytic Ricker sample values,
  including arrivals between sample centers.
- [ ] T2 water-bottom multiples use water velocity at every order, not the
  primary RMS velocity at the same T0.
- [ ] T3 interbed and coincident peg-leg interference, all variant switches.
- [ ] x=0 agrees with normal-incidence reflectivity under the selected R/T-only
  convention, within explicit path/FFT truncation tolerances.
- [ ] Path sums match selected omega-p responses as described in section 6.
- [ ] Offset batching/order invariance and negative-offset symmetry.
- [ ] Unsupported offsets, empty catalogs, resource limits, and bad inputs have
  explicit behavior; no silent fallback to the slant stack.
- [ ] No unexplained event-free-window energy beyond analytic wavelet tails
  and the declared numerical/truncation tolerance.

Avoid adding arbitrary spreading options in the first implementation. If physical
source normalization is requested, stop and make that a separate design task.

## 8. P2: semblance, figures, and stronger artifact checks

### Semblance

- Keep raw noiseless gathers as the default validation data. Make the existing
  seeded 1% noise floor optional and explicitly labelled, not an artifact fix.
- The current `scan_slowness_bound` includes the old synthesis taper. Do not
  apply that taper-derived restriction blindly to the ray backend. Preserve
  actual ray-support limits and make the distinction explicit.
- Report contributing trace counts/fold or validity alongside semblance so
  low-support high coherence is not mistaken for a well-resolved event.
- Reassess the documented shallow-interface velocity bias using clean gathers.
  Do not promise it disappears: finite spread, wavelet bandwidth, the time gate,
  interference, and scan-cell-dependent support also affect picks.
- Keep small-spread RMS velocity anchors separate from finite-offset best-fit
  stacking velocities; exact layered moveout is generally nonhyperbolic.
- Preserve the independent hand-built-hyperbola tests. Do not validate a
  catalog-generated gather only against overlays from the same catalog.

### Artifact measurements

The old `test_offset_gather_causality_and_taper` measures leakage before
`x/v_max - 0.04 s`, normalized by the strongest peak of the whole gather.
It misses much near-offset/deep-event smearing and can hide contamination as
strong as a weak interbed multiple. Add tests for:

- Event-free windows between isolated arrivals, at near/mid/far offsets.
- Leakage relative to local event amplitudes and the weakest target multiple,
  as well as a global reference amplitude.
- Waveform/peak errors, timing errors, and refinement convergence separately.
- All variants and canonical models, plus a simple model with a deliberately
  isolated weak multiple if the current T3 events overlap too strongly.

Use analytic wavelet-tail margins, not an assertion of zero energy at every
sample before an arrival. For the old backend, document irreducible finite-
aperture distortion rather than weakening tests or claiming elimination.

### Figures and documentation

- Add explicit backend selection in demos without silently changing the library
  default. Keep comparison outputs under distinct deterministic filenames.
- Plot only variant-appropriate event overlays, or clearly mark excluded-family
  references. Existing primaries-only figures show multiple overlays too.
- Shorten/wrap titles: existing offset figures clip the amplitude-convention text.
- Use shared, documented display scales for comparisons; show enlarged weak-event
  and near-offset windows. Do not pass visual acceptance through clipping alone.
- Update README: it still says “M0 scaffold”.
- Update PLAN: add the P0 corrections and new diagnostic backend stage, split M6
  into enumeration/synthesis versus interpretation/overlays, and mark work done
  only after its acceptance tests pass. Preserve the existing uncommitted edits.
- Reconcile “zero-offset preserved” with the old backend's different waveform
  and amplitude convention. Replace “event times stay exact” near aperture edges
  with qualified statements distinguishing ray/group times from measured peaks;
  PLAN already documents edge-related shifts of about 10 ms.
- Remove the blanket claim that all remaining near-offset smearing is unavoidable
  before rechecking it with converged quadrature. Some of it may be numerical.

## 9. Suggested execution order and completion gates

Implement in small checkpoints, not a single large rewrite:

1. P0-A quadrature regression tests and fix.
2. P0-B sampling/convergence regression tests and fix.
3. Confirm the additive ray API/amplitude contract; implement event geometry and
   coefficients with hand-built paths.
4. Implement bounded enumeration and independent reflectivity comparisons.
5. Implement diagnostic gathers and analytic artifact tests.
6. Integrate backend selection, semblance support handling, and figures.
7. Update PLAN/README/conventions; record remaining limitations and timings.

For each checkpoint, use task tracking, inspect the diff, and run:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Format only changed Python files with `uv run ruff format <paths>`; avoid
unrelated churn in the existing user changes. Before finishing figure-producing
work, generate the relevant demos with `--no-show` and inspect the images. If the
active model cannot view images, use the multimodal delegation procedure in
AGENTS.md. Ask specific questions about faint pre-event streaks, weak multiples,
overlay alignment, and clipped titles, not just “does this look OK?”.

Final report must state: files changed, tests passed, measured improvements,
backend/amplitude conventions, path bounds and convergence evidence, unsupported
aperture regions, and unresolved limits. Do not call a capped/truncated result
“complete” or an unvalidated gather “artifact-free”.
