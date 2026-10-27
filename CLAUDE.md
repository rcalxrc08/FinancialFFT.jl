# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FinancialFFT.jl is a Julia package that prices European options (vanilla, binary, and smile variants) via Fourier/characteristic-function methods. It is an extension of [FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl): it reuses that package's processes (`BlackScholesProcess`, `MertonProcess`, `KouProcess`, NIG, VarianceGamma, …), `EuropeanOption`, `ZeroRate`, `Underlying`, and adds new `pricer` methods, so the only export is `pricer` plus the method configurator structs.

## Commands

```julia
] test FinancialFFT                      # full suite (what CI runs, Julia 1.11 / 1.12), ~7 minutes
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(test_args = ["extrema_computation", "multiple_options"])'  # selected suites
```

- Tests are organised in suites: `test/runtests.jl` loads the shared `utils.jl` and `cases.jl`, then `include`s `<suite>_test.jl` for every name in its `SUITES` list (or only those passed as `test_args`). A new suite file must be added to `SUITES`; other files in `test/` are not run.
  - `cases.jl` (`METHOD_CASES`) declares, per method, the supported payoffs and the tolerance for each number type / AD back-end; `black_scholes` and `gradients` loop over it against `blsprice`/`blsbin` from FinancialToolbox. `other_models` compares the other methods with `LewisMethod` on Merton, Kou, VG and NIG (`model_cases`).
  - Known failures are recorded as `Broken(toll)` (see `utils.jl`), which becomes `@test_broken`: the suite reports "Unexpected Pass" as soon as one starts working, so update `cases.jl` when fixing one.
- Formatting: JuliaFormatter with `.JuliaFormatter.toml` (margin=1200, i.e. effectively no line wrapping).
- Docs: `docs/make.jl` (Documenter). CI also uses a custom registry `rcalxrc08/LocalRegistry` for `FinancialMonteCarlo`; resolve it when setting up a fresh environment.

## Architecture

Entry point [src/FinancialFFT.jl](src/FinancialFFT.jl) includes, in order: `abstracts`, `extrema_computation_opt`, `utils`, `charexp`, then the methods (`carrmadan`, `carrmadan_lewis`, `lewis`, `cos`, `density`). Order matters (later files use earlier definitions).

**Dispatch design.** Each method is a configurator struct subtyping `FinancialMonteCarlo.AbstractMethod` via `AbstractFFTMethod` (Carr-Madan, Carr-Madan-Lewis; use `fft`) or `AbstractIntegralMethod` (Lewis, COS, DensityInverter; direct quadrature/cosine sums). Pricing is invoked as `pricer(process, zero_rate, method, payoff[, mode])`, extending `FinancialMonteCarlo.pricer`. Each method file typically has: a vector-of-strikes core implementation, an array-of-payoffs overload, and a single-option overload.

**Characteristic exponents** ([src/methods/charexp.jl](src/methods/charexp.jl)): per-process `characteristic_exponent_i(imv, process)` (scalar) and `characteristic_exponent_vi` (broadcast), implemented as real functions of an imaginary-argument input (`ψ(v) = ψ_i(v*im)`) to avoid complex arithmetic. A new process needs these methods. `charexp_and_ders.jl` (untracked) is WIP for derivatives.

**Payoff handling** ([src/methods/utils.jl](src/methods/utils.jl), `smile.jl`): everything is priced as a call and converted with `call_to_put` (dispatching on vanilla/binary/smile option types).

**Truncation range of the COS method** ([src/methods/extrema_computation_opt.jl](src/methods/extrema_computation_opt.jl)): `[a, b]` comes from a Chernoff bound on the tails of the log-price, `t(s) = (T ψ_i(mul s) - log ε) / s` minimised over `s` by bisection on its (complex-step) derivative. The search bracket starts at `[1, 50]` and is doubled for short maturities (the optimum grows like `1/√T`), but never beyond `exponential_moment_bound(process, mul)`, where `E[exp(mul s X)]` stops being finite (Kou `λ₊`/`λ₋`, VG/NIG from the log/sqrt argument, `Inf` otherwise): past it `characteristic_exponent_i` returns meaningless finite values. A new process with semi-heavy tails needs an `exponential_moment_bound` method. The range is centered on the forward and shared by all strikes, so COS is inaccurate when `|log(S0/K)|` exceeds its half width (deep ITM/OTM at very short maturities; tracked as `Broken` in `multiple_options_test.jl`).

**Execution modes / optional backends.** `FinancialMonteCarlo.BaseMode` (`SerialMode`, `AbstractCudaMode`) is the trailing `pricer` arg. Optional packages are package extensions (`[weakdeps]`/`[extensions]` in `Project.toml`, code in [ext/](ext/)): CUDA (GPU Carr-Madan, plus `adapt_array`/`adapt_itp` hooks that are identity by default), DualNumbers, HyperDualNumbers, ForwardDiff, TaylorSeries (these supply `fft` overloads and `real_mod`/`imag_mod`/`v_value_mod` hooks so AD works through FFT and the range computation). The core code calls hooks like `real_mod`, `imag_mod`, `adapt_array` instead of `real`/`imag`/arrays directly; use them when editing numeric kernels. `ChainRulesCore.@ignore_derivatives` is used around non-differentiable setup (pattern vectors, asserts) for Zygote/ReverseDiff support.

## Working-tree caveats

The working tree contains many untracked scratch files (`* copy.jl`, `* - Copia.jl`, `*_old.jl`, `zygote_mwe.jl`, `pluto/`, `runner_cuda.jl`). Only files `include`d in `src/FinancialFFT.jl` are live code; do not edit the copies. Note `cosc.jl`, `density_cos.jl`, `extrema_computation.jl` are also not included by the module.
