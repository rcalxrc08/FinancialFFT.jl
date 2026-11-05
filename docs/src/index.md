```@raw html
<img class="pkg-hero-logo" src="assets/logo.png" alt="FinancialFFT.jl logo" width="96" height="96">
```

# FinancialFFT.jl

```@raw html
<p class="pkg-roles"><b>Fourier pricing</b> · <b>Densities</b> · <b>Automatic differentiation</b> · <b>GPU</b></p>
<div class="pkg-intro">
<p>FinancialFFT.jl prices European options from the characteristic function of the log-price. A model enters only
through its characteristic exponent, so pricing Merton, Kou, Variance Gamma, NIG or Heston costs about as much as Black
and Scholes: a quadrature, a cosine series or an FFT, with no simulation noise. Prices are generic in the number
type, so every Greek comes from automatic differentiation, and the same code runs on the GPU.</p>
</div>
<nav class="pkg-links">
<a class="pkg-btn" href="https://github.com/rcalxrc08/FinancialFFT.jl"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg> GitHub</a>
<a class="pkg-btn" href="https://github.com/rcalxrc08/FinancialFFT.jl/stargazers">★ Star</a>
<a class="pkg-btn" href="#Quick-start">Get started ↓</a>
<a class="pkg-btn" href="https://rcalxrc08.github.io/">All projects</a>
</nav>
```

![Implied volatility smiles of six models](assets/figures/smiles-light.svg)![Implied volatility smiles of six models](assets/figures/smiles-dark.svg)

## Relation to FinancialMonteCarlo.jl

The package adds methods to the `pricer` of [FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl),
and uses its types for everything else: the model (`BlackScholesProcess`, `HestonProcess`, …), the underlying
(`Underlying`), the curve (`ZeroRate`) and the payoff (`EuropeanOption`, `BinaryEuropeanOption`, and their smiles).
The last-but-one argument picks the numerical method, so moving a European option from simulation to a Fourier method
changes one argument:

```julia
pricer(model, zero_rate, MonteCarloConfiguration(10^6, 1), option)  # Monte Carlo, FinancialMonteCarlo.jl
pricer(model, zero_rate, CosMethod(2^12), option)                    # COS method, FinancialFFT.jl
```

## Features

- Pricing methods: [`LewisMethod`](@ref), [`CosMethod`](@ref), [`CarrMadanLewisMethod`](@ref), [`CarrMadanMethod`](@ref), [`CosDensityMethod`](@ref) and [`FFTDensityInverter`](@ref), compared in [Methods](methods.md).
- Densities of the log-return through [`density`](@ref), for the likelihood ratio method and Vibrato: [`CosDensityMethod`](@ref) and [`FFTDensityInverter`](@ref), see [Densities](methods.md#Densities).
- Models: Black–Scholes, Merton, Kou, Variance Gamma, Normal Inverse Gaussian, Heston.
- Payoffs: European and binary (digital) options, calls and puts, and their smiles (several strikes, one maturity), with every method. Options with different maturities are priced by broadcasting, `pricer.(model, zero_rate, method, options)`.
- Non-constant zero rates and dividends.
- Automatic differentiation with DualNumbers.jl, HyperDualNumbers.jl, ForwardDiff.jl, ReverseDiff.jl, Zygote.jl, Mooncake.jl and TaylorSeries.jl, see [Automatic differentiation](@ref) for the exceptions.
- Float32 and BigFloat prices, see [Floating-point types](@ref).
- GPU execution through CUDA.jl, see [GPU](@ref).

## Installation

```julia
] add FinancialFFT
```

## Quick start

```@example quickstart
using FinancialFFT, FinancialMonteCarlo

S0, K, r, d, T, σ = 100.0, 100.0, 0.02, 0.01, 1.0, 0.2

model = BlackScholesProcess(σ, Underlying(S0, d))
zero_rate = ZeroRate(r)
option = EuropeanOption(T, K)

pricer(model, zero_rate, LewisMethod(600.0, 20000), option)
```

The same call works with any other method and model:

```@example quickstart
model = MertonProcess(σ, 5.0, 0.03, 0.02, Underlying(S0, d))
pricer(model, zero_rate, CosMethod(2^12), option)
```

## GPU

With CUDA.jl loaded, passing `FinancialMonteCarlo.CudaMode()` as the last argument of `pricer` evaluates the
characteristic function and the integrals on the GPU:

```julia
using CUDA
smile = FinancialMonteCarlo.EuropeanOptionSmile(T, CuArray([90.0, 100.0, 110.0]), CuArray([true, true, false]))
pricer(model, zero_rate, CosMethod(2^12), smile, FinancialMonteCarlo.CudaMode())
```

- Every method runs on the GPU, with the accuracy it has on the CPU.
- `density(model, zero_rate, method, T, FinancialMonteCarlo.CudaMode())` computes the density on the GPU with `FFTDensityInverter` and `CosDensityMethod`, and [`FinancialFFT.density_values`](@ref) evaluates it there at a `CuArray` of points, differentiable with ForwardDiff.jl and Zygote.jl.
- The strikes of a smile must be on the device. A single option is given as on the CPU.
- ForwardDiff.jl and Zygote.jl differentiate the prices on the GPU. ReverseDiff.jl and Mooncake.jl are not supported on the GPU.

## Testing

```julia
] test FinancialFFT
```

The GPU tests are not part of the default run; on a machine with a CUDA GPU they run with the following command, which adds CUDA.jl to the test environment:

```julia
using Pkg
Pkg.test("FinancialFFT"; test_args = ["gpu"])
```

## Contents

```@contents
Pages = ["methods.md", "payoffs.md", "models.md", "ad.md", "api.md"]
Depth = 2
```
