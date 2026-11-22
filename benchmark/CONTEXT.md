# Benchmarks

The BenchmarkTools suite (Benchmark workflow) and the AD tables of the docs.

## Inputs

| Load | Why |
|:--|:--|
| `benchmarks.jl` | the suite, built in parts: Black Scholes per method, models, densities, the Chernoff range in Float64/ForwardDiff/TaylorN |
| `run.jl` | one part per process: `julia benchmark/run.jl OUT.json [PACKAGE_DIR]` |
| `compare.jl` | `julia benchmark/compare.jl HEAD.json [BASE.json]` |
| `ad.jl` | every back-end, every method, the Heston tasks, as multiples of a price (`docs/src/ad.md`) |
| `ad_gpu.jl` | ForwardDiff and Zygote in `CudaMode` (needs CUDA in the environment) |
| `references/performance.md` | where the numbers live, how to measure |

## Process

- The benchmark `[compat]` follows the package's (FinancialMonteCarlo).
- A hot-loop change is timed before and after, old commit in a `git worktree`, fresh processes, idle machine.

## Outputs

Benchmark workflow summaries and artifacts; the tables of `docs/src/ad.md`.
