# Stage 03: shared utilities

Grids and float types, the FFT functions, Clenshaw sums, interpolation at the strikes, payoff conversions.

## Inputs

| Load | Why |
|:--|:--|
| `utils.jl` | `adapt_array`, `zero_promoted`, `float_type`, `to_float_type`, `brfft_mod`, `brfft_adjoint`, `plan_fft_mod!`/`fft_mod!`, `InPlaceFFTPlan`, `clenshaw_exp_sum`, `clenshaw_cos_sum` and their `rrule`s, `grid_interpolate`, `lagrange_quintic`, `call_to_put`, `put_to_option` |
| `references/fft_utils.md` | the FFT plans per back-end, float types and FFTA, Clenshaw and Reinsch, interpolation, payoff conversions |
| `references/autodiff.md` | which back-end overloads `brfft_mod` and `fft` |
| `references/gpu.md` | device arrays, no element-wise indexing |

## Process

- A change to `brfft_mod` or `fft` touches every AD extension: keep their methods in step.
- Clenshaw: keep Reinsch's form near θ = 0, π; the `rrule`s are for Zygote only (O(N²) without them).
- Interpolation with gathers and scatters only (device, every back-end).

## Outputs

Shared by stages 05 and 06. Checked by `test/types_test.jl` (float types, BigFloat via FFTA), `gradients`,
`multiple_options` (Clenshaw for smiles), `gpu`.

## Verify (Kaimon)

Setup: `using FinancialFFT`.

| Check | Expect |
|:--|:--|
| `FinancialFFT.float_type(1f0, 2)` | `Float32` |
| `run_tests()` with `test_args = ["types", "multiple_options"]` | pass |
