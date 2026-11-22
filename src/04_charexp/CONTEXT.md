# Stage 04: characteristic exponents

The characteristic exponent of every supported FinancialMonteCarlo process.

## Inputs

| Load | Why |
|:--|:--|
| `charexp.jl` | `characteristic_exponent_i(imv, process[, T])`, `characteristic_exponent_vi`, the `Base.broadcasted` overload, its `rrule` |
| `references/charexp.md` | the imaginary-argument convention, the broadcast loop, Lévy vs Heston, open issues |
| `references/range.md` | a new process with semi-heavy tails also needs `exponential_moment_bound` |
| `references/autodiff.md` | `sqrt_mod`, hooks |

## Process

- A new process: `characteristic_exponent_i`/`_vi` (three-argument if not Lévy), `exponential_moment_bound`, a row in
  `model_cases` of the tests, `docs/src/models.md`.
- Complex `sqrt` through `sqrt_mod`; real functions of an imaginary input where possible.
- `charexp_and_ders.jl` here is untracked and git-excluded WIP, not included by the module.

## Outputs

`ψ_i` for Black–Scholes, Merton, Kou, VG, NIG, Heston. Checked by `test/other_models_test.jl` (against `LewisMethod`)
and `test/black_scholes_test.jl`.

## Verify (Kaimon)

Setup: `using FinancialFFT, FinancialMonteCarlo`; `bs = BlackScholesProcess(0.2, Underlying(100.0, 0.0))`.

| Check | Expect |
|:--|:--|
| `FinancialFFT.characteristic_exponent_i(1.0, bs)` | `0.02` (σ²/2: the drift correction is applied by the methods) |
| `run_tests()` with `test_args = ["other_models"]` | pass |
