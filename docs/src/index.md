# FinancialFFT.jl

FinancialFFT.jl prices European options with Fourier methods: every price is computed from the
characteristic function of the log-price, so any model with a known characteristic exponent can be priced
in closed form up to a numerical integration.

The package extends [FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl):
models (`BlackScholesProcess`, `MertonProcess`, …), curves (`ZeroRate`), underlyings (`Underlying`) and
payoffs (`EuropeanOption`, `BinaryEuropeanOption`, …) are the ones defined there, and FinancialFFT adds new
methods to `pricer`. Switching from Monte Carlo to a Fourier method only means replacing the
`MonteCarloConfiguration` with one of the methods below.

## Features

- Methods: [`LewisMethod`](@ref), [`CosMethod`](@ref), [`CarrMadanLewisMethod`](@ref), [`CarrMadanMethod`](@ref), [`FinancialFFT.DensityInverter`](@ref).
- Models: Black–Scholes, Merton, Kou, Variance Gamma, Normal Inverse Gaussian.
- Payoffs: European and binary (digital) options, calls and puts, smiles and arrays of options with different maturities.
- Non-constant zero rates and dividends.
- Automatic differentiation with DualNumbers.jl, HyperDualNumbers.jl, ForwardDiff.jl, ReverseDiff.jl, Zygote.jl, Mooncake.jl and TaylorSeries.jl (Zygote gradients of `CarrMadanMethod` and `CarrMadanLewisMethod` are not reliable, since the spline is excluded from differentiation; ReverseDiff.jl requires version 1.18 or later).
- GPU execution through CUDA.jl (experimental).

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

## Testing

```julia
] test FinancialFFT
```

## Contents

```@contents
Pages = ["methods.md", "payoffs.md", "models.md", "ad.md", "api.md"]
Depth = 2
```
