# CLAUDE.md

**FinancialFFT.jl** prices European options (vanilla, binary, smiles) and computes densities of the log-return with
Fourier methods (Carr-Madan, Carr-Madan-Lewis, Lewis, COS, FFT and COS densities). It extends FinancialMonteCarlo.jl:
processes, options, `ZeroRate`, `Underlying` and execution modes come from there, and the package adds `pricer` and
`density` methods, so the exports are `pricer`, `density` and the method structs. Quadrature weights come from
PatternVectors (private registry `rcalxrc08/LocalRegistry`), closed-form references from FinancialToolbox. Goals and
non-goals: `references/mission.md`; why things are the way they are: `references/decisions.md`.

This workspace follows the Interpretable Context Methodology (layered context, arXiv:2603.16021): load only what the
task needs.

| Layer | Where | What |
|:--|:--|:--|
| 0 | this file | identity, map, rules that hold everywhere |
| 1 | `CONTEXT.md` | which stage or folder a task belongs to, shared references |
| 2 | `src/0N_*/CONTEXT.md`, `ext/`, `test/`, `docs/`, `benchmark/`, `etc/` | a contract per stage: inputs, process, outputs, runtime checks |
| 3 | `references/*.md` | stable facts, decisions and measurements, loaded when a contract lists them |
| 4 | `state/STATUS.md` | what is in flight: current focus, open issues, known problems |

Startup: read `state/STATUS.md`, then route the task through `CONTEXT.md`. Runtime truth comes from a Kaimon session
for this project (`kaimon.toml` pins the Julia); each contract's Verify table lists the checks to run after a change.

Map: `src/FinancialFFT.jl` includes the stages in order (later files use earlier definitions), `01_hooks` → `02_range` →
`03_utils` → `04_charexp` → `05_methods` → `06_densities`; `ext/` one extension per AD back-end and CUDA; `test/` the
suites; `docs/` the Documenter site; `benchmark/` the suite and the AD tables; `examples/` the getting-started script;
`etc/` the logo. Only included files are live code: untracked copies are excluded (`references/workflows.md`).

Rules for every change (details in `references/conventions.md` and `references/autodiff.md`):

- Prices keep the float type of the inputs: no Float64 literals, no `pi * N`, no float ranges (Enzyme).
- Numeric kernels go through the hooks (`real_mod`, `v_value_mod`, `ignore_derivatives_mod`, `adapt_array`, …);
  constants for AD inside `@ignore_derivatives ignore_derivatives_mod.(…)`.
- No element-wise indexing of arrays that may live on the device; pricing paths type stable.
- Every quadrature names its rule and order; `@muladd` for every `a * b + c`.
- Invalid arguments throw `DomainError`/`ArgumentError` through `src/01_hooks/checks.jl`, never `@assert` (#89).
- Reuse the author's packages and established libraries before writing new code.
- Format with `.JuliaFormatter.toml`; update the CONTEXT.md, references and `state/STATUS.md` a change touches in the
  same commit (a new rule also gets an entry in `references/decisions.md`); default branch `master`; push to `origin`.
