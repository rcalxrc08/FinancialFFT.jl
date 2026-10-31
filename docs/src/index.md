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
- Models: Black–Scholes, Merton, Kou, Variance Gamma, Normal Inverse Gaussian, Heston.
- Payoffs: European and binary (digital) options, calls and puts, and their smiles (several strikes, one maturity), with every method. Options with different maturities are priced by broadcasting, `pricer.(model, zero_rate, method, options)`.
- Non-constant zero rates and dividends.
- Automatic differentiation with DualNumbers.jl, HyperDualNumbers.jl, ForwardDiff.jl, ReverseDiff.jl, Zygote.jl, Mooncake.jl and TaylorSeries.jl (Zygote gradients of `CarrMadanMethod` and `CarrMadanLewisMethod` are not reliable, since the spline is excluded from differentiation; ReverseDiff.jl requires version 1.18 or later).
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

- `LewisMethod`, `CosMethod`, `CarrMadanLewisMethod` and `DensityInverter` run on the GPU, with the accuracy they have on the CPU. `CarrMadanMethod` ignores the mode and runs on the CPU.
- The strikes of a smile must be on the device. A single option is given as on the CPU.
- ForwardDiff.jl and Zygote.jl differentiate the prices on the GPU, with the same limitations as on the CPU. ReverseDiff.jl and Mooncake.jl are not supported on the GPU.

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
