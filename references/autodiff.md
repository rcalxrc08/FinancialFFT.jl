# Automatic differentiation

_The hooks, what each back-end needs, the upstream issues._ (Layer 3 reference; GPU-specific limits in `gpu.md`.)

## Hooks (`src/01_hooks/mods.jl`)

- The core code calls the scalar hooks of `mods.jl` (`real_mod`, `imag_mod`, `exp_mod`, `sincos_mod`, `sqrt_mod`,
  `v_value_mod`, `ignore_derivatives_mod`), `zero_promoted` and `adapt_array` (`src/03_utils/utils.jl`) instead of
  `real`/`imag`/arrays directly; use them when editing numeric kernels.
- `ChainRulesCore.@ignore_derivatives` is used around non-differentiable setup (pattern vectors, checks) for Zygote.
  ReverseDiff, Mooncake and Enzyme ignore `@ignore_derivatives`: wrap quantities that must stay constants (truncation
  ranges, grid bounds) in `ignore_derivatives_mod` inside the macro (`@ignore_derivatives ignore_derivatives_mod.(…)`),
  or they become tracked and break integer conversions. The hook is `v_value_mod` (dual numbers, Taylor series, tracked
  reals), a zero-derivative primitive in the Mooncake extension; the macro is still needed for Zygote, and keeps it from
  tracing the computation.
- A variable assigned both inside a closure of `@ignore_derivatives` and outside it is boxed (`Core.Box`) and hides
  every type after it: give it another name inside (`grid_interpolate`, `density_array` of `FFTDensity`).

## Number types (extensions)

- DualNumbers, HyperDualNumbers, ForwardDiff, TaylorSeries supply `fft` overloads and `real_mod`/`imag_mod`/
  `v_value_mod` hooks so AD works through the FFT and the range computation. The ForwardDiff `fft!` uses only
  broadcasts and one batched `fft!` with the components along a new last dimension (nested duals of a Hessian recurse
  into it, one dimension per level).
- TaylorSeries covers `Taylor1` and `TaylorN`, for which there is no `cis` (JuliaDiff/TaylorSeries.jl#445): the kernels
  fold phases into a complex `exp` instead. Its element type says neither the order nor the `JetSpace`, so build
  constants from a value, with `zero(x)`, not from a type; the zero of a common type comes from `zero_promoted(x...)`,
  which the extension builds from the values for series. #51: TaylorSeries 0.23 warns when the TaylorN variables change
  the default JetSpace.
- DualNumbers and HyperDualNumbers have no `sincos`, hence `sincos_mod`, overloaded with one `sincos` of the value.

## ReverseDiff

- `fft` recorded as one tape instruction with a `bfft` pullback, `v_value_mod` for tracked reals; needs ReverseDiff ≥
  1.18 (1.17 overflows the stack materializing complex broadcasts of tracked values), and PatternVectors ≥ 0.1.4
  (broadcast style conflict with `TrackedStyle`).
- ReverseDiff cannot broadcast a function that captures tracked values: pass them as broadcast arguments, or use `map`.

## Mooncake (0.5)

- `fft`/`brfft_mod` are primitives with `frule!!`/`rrule!!` (Mooncake cannot differentiate the FFTW plan);
  `ignore_derivatives_mod` and `v_value_mod` are zero-derivative primitives.
- Mooncake fails on type-unstable code with small `Union` return types (e.g. `Union{Float16, Float64}`): keep pricing
  paths type stable.
- It cannot differentiate the complex `sqrt` of Base (it bitcasts): use `sqrt_mod`, overlaid in the extension for
  `Complex{<:Real}`, not only IEEE floats, since Mooncake over ForwardDiff (FinancialCalibrator's nested implicit
  derivatives) meets `Complex{<:Dual}` there (FinancialCalibrator.jl#41, released in 0.6.3).

## Zygote

- ChainRules rules: `brfft_mod` (AbstractFFTs' rule builds its weights on the CPU), the Clenshaw sums (#47), the
  vectorised characteristic exponent (#77).
- PatternVectors ≥ 0.1.6 does `dot` with broadcasts (0.1.7 without ambiguities with ReverseDiff); ≥ 0.1.8 unthunks the
  deferred cotangents of Zygote ≥ 0.7.17, which made the cotangent of `sum` a host `Vector` and broke every Zygote GPU
  gradient but Lewis (#87, PatternVectors#19); the tests require Zygote 0.7.18. The compat is 0.1.9 (also the
  constructor error out of line, without which Enzyme ≥ 0.13.219 aborted, EnzymeAD/Enzyme.jl#4053, #86).
- Traps found while profiling FinancialCalibrator (Zygote 0.7.18): `oftype` has an uninferred pullback
  (FluxML/Zygote.jl#1751, use `promote`/`convert`); `gradient`/`pullback` are compiled for `f::Function` (#1752); every
  closure capture traces `Core._typeof_captured_variable` (#1753); branch-only code keeps `Vector{Any}` stacks (#1754:
  FinancialFFT's `pricer` has 7 of them on the Lewis path).

## Enzyme (#62, `ext/FinancialFFTEnzymeCoreExt.jl`)

- Supported on the CPU, every method and density, forward and reverse (suite `enzyme`, which loads Enzyme itself;
  tolerance `enzyme` in `cases.jl`), through an extension on EnzymeCore, never through Enzyme-specific core code.
- The extension marks inactive the hooks (`v_value_mod`, `ignore_derivatives_mod`, whose array method exists for it:
  Enzyme cannot prove a broadcast of the hook over an active array constant) and the setup functions
  (`chernoff_optimum`, `exponential_moment_bound`, `fft_grid`, `carr_madan_damping`, `carr_madan_weights`,
  `density_integration_grid`), and gives forward and reverse rules (batched) for `fft` and `brfft_mod`. Keep constant
  setup in named functions that it can mark.
- Enzyme fails on float ranges even of constants (TwicePrecision bit masks, EnzymeAD/Enzyme.jl#4016): the grids are
  comprehension vectors (`k * dx`, `k / M`, rounded as the ranges were; the COS nodes `k * π / (b - a)` too since
  FinancialCalibrator.jl#41, where `(0:N) * (π / bma)` stopped `calibrate` with Enzyme on Heston with
  `EnzymeNoDerivativeError` in `floatrange`).
- In `Float32` Enzyme is wrong without an error on every method with a complex `sqrt` or `inv` (Heston): Enzyme 0.13.221
  maps Base's `sqrt`/`inv` of `ComplexF32` to intrinsics with a zero derivative, and a `ComplexF32` kept on the tape of a
  custom rule comes back corrupted, so a rule on `sqrt_mod` does not fix reverse mode (EnzymeAD/Enzyme.jl#4108, tried
  and reverted 2026-11-21); `ComplexF64` is right.
- It also fails on `map` over a constant array storing active values (`density_values` of a `CosDensity` broadcasts
  `density_value.(f, y)`: `Base.broadcastable` of an `AbstractDensity` is a `Ref`, so no `Ref` at the call), on splatted
  calls `x -> f(x...)` and on payoff types chosen at run time (the suite's `enzyme_bs_pricer`).
- Models on a dividend curve (`Underlying(S0, curve)`, an `UnderlyingVec`, what FinancialCalibrator's `adapter`
  builds) need `set_runtime_activity` in reverse mode: the `Curve` holds `Vector`s next to the active parameters
  (rcalxrc08/FinancialMonteCarlo.jl#79, Enzyme 0.13.221; the tests use scalar dividends).
