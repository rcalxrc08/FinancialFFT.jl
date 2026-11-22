# Workflows: commands, Kaimon, CI, releases

_How to run, test, benchmark, document and release._ (Layer 3 reference.)

## Kaimon session

- `start_session(project_path = "C:/Users/Nicola/.julia/dev/FinancialFFT")`; `kaimon.toml` pins Julia 1.13.
- First call: `ex(q = false, e = "(VERSION, Base.active_project(), pathof(FinancialFFT))")`.
- The AD back-ends are test dependencies: check them through `run_tests()` (or `Pkg.test(test_args = [...])` in a
  separate process, suites take minutes).
- Do not run `julia --project` scripts that precompile this package while the session is open (Revise keeps the old
  code); restart the session afterwards.

## Commands

```julia
] test FinancialFFT                      # full suite (what CI runs, Julia 1.11 / 1.12)
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(test_args = ["extrema_computation", "multiple_options"])'  # selected suites
julia --project=. -e 'using Pkg; Pkg.test(test_args = ["gpu"])'  # GPU suite, needs a CUDA GPU (not in SUITES, not run by CI)
julia benchmark/run.jl OUT.json [PACKAGE_DIR]   # the BenchmarkTools suite, one part per process
julia benchmark/compare.jl HEAD.json [BASE.json]
julia --project=docs docs/make.jl
julia -e 'using JuliaFormatter; format(".")'   # .JuliaFormatter.toml (margin=1200)
```

## CI

`CI.yml`: Julia 1.11 and 1.12, ubuntu. `Benchmark.yml` compares a pull request with its base, reports master alone on
push, and compares with any ref on demand: `gh workflow run Benchmark.yml --ref master -f base=v0.5.0` (results in the
job summary and the `benchmarks` artifact). `DeployPage.yml` (docs), `TagBot.yml`, `CompatHelper.yml`. CI uses the
custom registry `rcalxrc08/LocalRegistry` for `PatternVectors`, a dependency registered only there (FinancialMonteCarlo
is in General); add it when setting up a fresh environment. Default branch `master`.

## Sub-environments

- FinancialMonteCarlo compat `"0.7, 0.8, 0.9"` in the package, tests, docs and benchmarks; the benchmark `[compat]`
  must follow the package's.
- `benchmark/` and `docs/` load the package through `[sources]` with a path; each has a `[compat]` entry for the package
  itself, so CompatHelper does not open PRs to add one. Bump these entries in the same commit as a breaking version bump,
  otherwise the environment fails to resolve.

## Release

Bump `version`, push to `master`, wait for CI, register (General), then make sure the tag `vX.Y.Z` and the GitHub
release exist (TagBot).

## Working tree

Untracked scratch files are excluded locally (`.git/info/exclude`: `src/*/* copy*.jl`, `*_old.jl`, `zygote_mwe.jl`,
`charexp_and_ders.jl`, `pluto/`, …). Only files `include`d in `src/FinancialFFT.jl` are live code; do not edit the copies.
