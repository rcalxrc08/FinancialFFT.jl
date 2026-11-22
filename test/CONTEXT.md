# Tests and CI

The suites, how to select them, what CI runs.

## Inputs

| Load | Why |
|:--|:--|
| `runtests.jl` | `SUITES`, selection by `test_args`, per-suite logging, CUDA added only for `gpu` |
| `utils.jl` | `Broken(toll)` and helpers |
| `cases.jl` | `METHOD_CASES` (payoffs and tolerances per method and back-end), `model_cases`, `mooncake_gpu` |
| `extrema_computation_test.jl` | the Chernoff range |
| `fft_parameters_test.jl` | grids from a tolerance; parameters constant on every back-end |
| `black_scholes_test.jl` | every method against FinancialToolbox |
| `gradients_test.jl` | every back-end against the closed-form Greeks |
| `enzyme_test.jl` | Enzyme forward and reverse (loads Enzyme itself) |
| `other_models_test.jl` | Merton, Kou, VG, NIG, Heston against `LewisMethod` |
| `multiple_options_test.jl` | smiles |
| `density_test.jl` | densities and their prices |
| `types_test.jl` | `@inferred`, float types, BigFloat |
| `aqua_test.jl` | Aqua, the one excluded ambiguity |
| `gpu_test.jl` | `CudaMode`, opt-in (`test_args = ["gpu"]`) |
| `context.jl` | the context layers follow the code (run before the suites) |
| `references/testing.md` | the case tables, `Broken`, durations |
| `references/workflows.md` | commands, CI matrix |

## Process

- A new suite file is added to `SUITES`; a known failure is a `Broken(toll)` in `cases.jl` (it reports "Unexpected Pass"
  when fixed).
- Select suites with `test_args` in the fast loop; the full suite takes ~50 minutes locally.

## Outputs

A green `Pkg.test()`; `context.jl` green whenever a file is added, moved or renamed.

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| `run_tests()` summary | pass; reports the Julia of `kaimon.toml` |
| `include("test/context.jl")` | pass |
