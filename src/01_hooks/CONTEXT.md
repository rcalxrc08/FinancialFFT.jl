# Stage 01: types and hooks

The method type hierarchy and the scalar hooks every kernel calls instead of `real`, `imag`, `exp`, `sincos`, `sqrt`.

## Inputs

| Load | Why |
|:--|:--|
| `abstracts.jl` | `AbstractIntegrationMethod`, `AbstractFFTMethod`, `AbstractIntegralMethod`, `AbstractDensityMethod` |
| `mods.jl` | `real_mod`, `imag_mod`, `exp_mod`, `sincos_mod`, `sqrt_mod`, `v_value_mod`, `ignore_derivatives_mod` |
| `checks.jl` | `check_domain` and one `@non_differentiable` validation per constructor (`check_fft_grid`, `check_lewis`, …) |
| `references/autodiff.md` | which extension overloads which hook and why |
| `references/conventions.md` | before adding a hook |

## Process

- A hook has a generic method here and specific methods in the extensions (zero-derivative primitive for Mooncake,
  inactive for Enzyme, component-wise for duals and series).
- `sqrt_mod` exists because Mooncake cannot differentiate Base's complex `sqrt`; `sincos_mod` because DualNumbers and
  HyperDualNumbers have no `sincos`.
- Constructors validate through one function each from `checks.jl` (`DomainError` with the value), marked
  `@non_differentiable` and called without `@ignore_derivatives`: its closure costs ~11 μs per block under Zygote
  (FinancialMonteCarlo#80). `@assert` only for internal invariants.

## Outputs

The hooks, used by every later stage. Checked by every suite through the back-ends (`gradients`, `enzyme`, `types`).

## Verify (Kaimon)

Setup: `using FinancialFFT`.

| Check | Expect |
|:--|:--|
| `FinancialFFT.v_value_mod(1.5)`, `FinancialFFT.real_mod(1.0 + 2.0im)` | `1.5`, `1.0` |
| `LewisMethod(100.0, 2)` | throws `DomainError` (value 2) |
| `run_tests()` with `test_args = ["types"]` | pass |
