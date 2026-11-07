# FinancialFFT.jl <img src="etc/logo.png" width="40">  

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rcalxrc08.github.io/FinancialFFT.jl/)
[![codecov](https://codecov.io/gh/rcalxrc08/FinancialFFT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/rcalxrc08/FinancialFFT.jl)
[![CI](https://github.com/rcalxrc08/FinancialFFT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/rcalxrc08/FinancialFFT.jl/actions/workflows/CI.yml)
##### This is a Julia package for pricing European options with Fourier methods. It extends [FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl) with new methods of `pricer`.

It currently contains the following capabilities:

- Support for the following Pricing Methods:
    - Lewis
    - COS
    - Carr-Madan
    - Carr-Madan-Lewis
    - COS density (`CosDensityMethod`)
    - FFT density inversion (`FFTDensityInverter`)
- Densities of the log-return, for the likelihood ratio method and Vibrato (`density`):
    - FFT inversion of the characteristic function (`FFTDensityInverter`)
    - COS expansion (`CosDensityMethod`)
- Support for the following Single Name Models:
    - Black Scholes
    - Heston
    - Kou
    - Merton
    - Normal Inverse Gaussian
    - Variance Gamma
- Support for non constant zero rates and dividends
- Support for the following Option families:
    - European Options
    - Binary European Options
    - Smiles of both (several strikes, one maturity)

Currently supports [DualNumbers.jl](https://github.com/JuliaDiff/DualNumbers.jl), [HyperDualNumbers.jl](https://github.com/JuliaDiff/HyperDualNumbers.jl), [TaylorSeries.jl](https://github.com/JuliaDiff/TaylorSeries.jl), [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl), [Zygote.jl](https://github.com/FluxML/Zygote.jl) and [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl)
for Automatic Differentiation (where it makes sense), see the [limitations](https://rcalxrc08.github.io/FinancialFFT.jl/ad/), and GPU execution through [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

## How to Install
The package and FinancialMonteCarlo.jl are in a custom registry. To install them simply type on the Julia REPL the following:
```julia
] registry add https://github.com/rcalxrc08/LocalRegistry.git
] add FinancialFFT, FinancialMonteCarlo
```
## How to Test
After the installation, to test the package type on the Julia REPL the following:
```julia
] test FinancialFFT
```
## Hello World: Pricing European Call Option in Black Scholes Model
The following example shows how to price a european call option with underlying varying according to the Black Scholes Model, given the volatility:
```julia
#Import the Packages
using FinancialFFT, FinancialMonteCarlo;

#Define Spot Datas
S0=100.0;
K=100.0;
r=0.02;
T=1.0;
d=0.01;
#Define Model Parameters
σ=0.2;
#Build the Structs
A = 600.0;
N = 18;
method_cm = CarrMadanMethod(A, N); #Configurator
zeroRate=ZeroRate(r);
underlying=Underlying(S0,d); #Underlying relative data

#Define The Option
EU_payoff=EuropeanOption(T,K)
#Define the Model
Model=BlackScholesProcess(σ,underlying);

#Price
@show EuPrice=pricer(Model,zeroRate,method_cm,EU_payoff);
```

## Execution modes
The execution mode is part of the method, as in FinancialMonteCarlo.jl (`mode` keyword of every constructor, default `SerialMode()`); with CUDA.jl loaded the same call runs on the GPU:
```julia
using CUDA
pricer(Model, zeroRate, CarrMadanMethod(A, N; mode = FinancialMonteCarlo.CudaMode()), EU_payoff)
```
`FinancialMonteCarlo.with_mode(method, mode)` gives a copy of a method in another mode.

## Curve Support
Non constant zero rates and dividend are managed.
An implied curve is built at time zero, such implied curve is able to return the right implied zero/dividend at a given time,
Without the need to carry the integral structure of the curve.
No support for multicurrency.