# Stage 05: pricing methods

The method structs and their `pricer`s: Carr-Madan, Carr-Madan-Lewis, parameters from a tolerance, Lewis, COS.

## Inputs

| Load | Why |
|:--|:--|
| `carrmadan.jl` | `CarrMadanMethod`, `FFTGrid`, `FFTTolerance`, `carr_madan_weights`, `carr_madan_damping`, the kernel |
| `carrmadan_lewis.jl` | `CarrMadanLewisMethod`, `lewis_denominator`, `carr_madan_lewis_call` |
| `fft_parameters.jl` | `fft_grid` for `FFTTolerance`: interpolation, truncation and aliasing bounds, `AliasingExponent`; `lewis_error_setup`, `lewis_aliasing_distance`, `truncation_frequency_bound` for Lewis |
| `lewis.jl` | `LewisMethod`, `QuadratureGrid`, `QuadratureTolerance`, `quadrature_grid`, `LewisMethodResult`, `lewis_coefficients`, `lewis_amplitudes_phases`, `lw_integrate` |
| `cos.jl` | `CosMethod`, `CosTerms`, `CosTolerance`, `cos_range_and_terms`, `cos_payoff_envelope`, `cos_phased_series`, `cos_clenshaw` (the GPU kernel `phased_cos_sum` is in `src/03_utils/utils.jl`) |
| `references/methods.md` | dispatch, quadratures, each method's design and measurements |
| `references/fft_utils.md` | interpolation, Clenshaw, payoff conversions |
| `references/gpu.md` | what runs on the device per back-end |
| `references/autodiff.md` | constants for AD, Enzyme limits |

## Process

- Four-argument `pricer` only; `mode` is a keyword of every public constructor and a positional field, so `with_mode`
  rebuilds the method.
- One `pricer` per method for options and smiles; no vectors of strikes or arrays of options.
- Grid parameters are constants for every back-end (`@ignore_derivatives` + `ignore_derivatives_mod`).
- `zygote_mwe.jl` here is untracked and git-excluded, not part of the package.

## Outputs

Prices in the float type of the inputs, smiles as vectors. Checked by `test/black_scholes_test.jl`,
`test/gradients_test.jl`, `test/fft_parameters_test.jl`, `test/other_models_test.jl`, `test/multiple_options_test.jl`,
`test/enzyme_test.jl`, `test/gpu_test.jl`.

## Verify (Kaimon)

Setup: `using FinancialFFT, FinancialMonteCarlo, FinancialToolbox`; `bs, und = BlackScholesProcess(0.2), Underlying(100.0, 0.0)`,
`r = ZeroRate(0.01)`, `opt = EuropeanOption(1.0, 100.0)`.

| Check | Expect |
|:--|:--|
| `pricer(bs, und, r, CosMethod(256), opt) - blsprice(100.0, 100.0, 0.01, 1.0, 0.2, 0.0)` | below 1e-10 |
| `@inferred pricer(bs, und, r, LewisMethod(400.0, 2^12), opt)` | `Float64` |
| `run_tests()` with `test_args = ["black_scholes", "fft_parameters"]` | pass |
