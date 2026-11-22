# Extensions

One extension per AD back-end and for CUDA; the package loads and works without any of them.

## Inputs

| Load | Why |
|:--|:--|
| `FinancialFFTCUDAExt.jl` | `grid_interpolate` of a number on the device |
| `FinancialFFTDualNumbersExt.jl` | `fft`/`brfft_mod` by components (one shared plan), hooks, `sincos_mod` |
| `FinancialFFTHyperDualNumbersExt.jl` | the same for hyper-duals |
| `FinancialFFTForwardDiffExt.jl` | one batched `fft!`/`brfft` with the partials along a new last dimension (nested duals recurse) |
| `FinancialFFTTaylorSeriesExt.jl` | `Taylor1`/`TaylorN`: components through one plan and buffers, `zero_promoted` from values |
| `FinancialFFTReverseDiffExt.jl` | `fft` and `brfft_mod` as tape instructions with cached plans, `v_value_mod` |
| `FinancialFFTMooncakeExt.jl` | `fft`/`brfft_mod` primitives, zero-derivative hooks, `sqrt_mod` overlay for `Complex{<:Real}` |
| `FinancialFFTEnzymeCoreExt.jl` | inactive hooks and setup functions, batched forward/reverse rules for `fft` and `brfft_mod` |
| `references/autodiff.md` | what each back-end needs and why |
| `references/fft_utils.md` | the FFT plans per back-end |
| `references/gpu.md` | the GPU limits per back-end |
| `references/dependencies.md` | compat floors and upstream workarounds |

## Process

- An extension adds methods to hooks and FFT functions; it never changes the meaning of a package method.
- Enzyme support only through EnzymeCore; new constant setup must be a named function marked inactive here.
- Upstream workarounds go to `references/dependencies.md` with the issue and version.

## Outputs

`Base.get_extension(FinancialFFT, :FinancialFFT<Dep>Ext) !== nothing` after `using <Dep>`. Checked by `gradients`,
`enzyme`, `types`, `gpu`.

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| `run_tests()` with `test_args = ["gradients", "enzyme"]` | pass (long: over 10 minutes each) |
