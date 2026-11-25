# Stage 06: densities

The `density` interface, `CosDensity`, `FFTDensity` and the pricer of every density method.

## Inputs

| Load | Why |
|:--|:--|
| `density.jl` | `density`, `AbstractDensity`, `density_values` |
| `utils_density.jl` | `mask_outside_range`, `density_integral_price`, `density_integration_grid` |
| `cos_density.jl` | `CosDensityMethod`, `CosDensity`, `density_value`/`density_array` |
| `fft_density.jl` | `FFTDensityInverter`, `FFTDensity`, `fft_density_values`, `lagrange_cubic` |
| `references/densities.md` | the interface, the two density types, the Hermitian FFT and its measurements |
| `references/range.md` | the range the densities live on |
| `references/autodiff.md` | Enzyme and the `Ref` broadcast of densities, Mooncake and moving grids |

## Process

- A density type implements `density_value` (the scalar kernel of `f(y)`) and `density_array` (required, no default,
  #78) without the mask; the interface applies it.
- A density method with an `M` field prices through `density_integral_price` (Simpson, put, parity).
- The grid of `FFTDensity` stays a constant for AD (`ignore_derivatives_mod`).
- VibratoMonteCarlo consumes `density` (its FFT density will be removed in favour of this one, Vibrato#37).

## Outputs

`AbstractDensity` objects and density prices. Checked by `test/density_test.jl`, `types`, `enzyme`, `gpu`.

## Verify (Kaimon)

Setup: `using FinancialFFT, FinancialMonteCarlo`; `bs, und = BlackScholesProcess(0.2), Underlying(100.0, 0.0)`,
`f = density(bs, und, ZeroRate(0.01), CosDensityMethod(), 1.0)`.

| Check | Expect |
|:--|:--|
| `f(0.0) > 0`, `f(10.0)` | `true`, `0` (outside the range) |
| `run_tests()` with `test_args = ["density"]` | pass |
