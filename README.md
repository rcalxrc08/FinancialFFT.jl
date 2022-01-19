# FinancialFFT.jl <img src="etc/logo.png" width="40">  

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rcalxrc08.gitlab.io/FinancialFFT.jl/)
[![Build Status](https://travis-ci.com/rcalxrc08/FinancialFFT.jl.svg?branch=master)](https://travis-ci.com/rcalxrc08/FinancialFFT.jl)[![pipeline status](https://gitlab.com/rcalxrc08/FinancialFFT.jl/badges/master/pipeline.svg)](https://gitlab.com/rcalxrc08/FinancialFFT.jl/commits/master) [![coverage report](https://gitlab.com/rcalxrc08/FinancialFFT.jl/badges/master/coverage.svg)](https://gitlab.com/rcalxrc08/FinancialFFT.jl/commits/master)
##### This is a Julia package containing some useful Financial function for Pricing and Risk Management for Equity products.

It currently contains the following capabilities:

- Support for the following Single Name Models:
    - Black Scholes
    - Kou
    - Merton
    - Normal Inverse Gaussian
    - Variance Gamma
- Support for non costant zero rates and dividends
- Support for the following Option families:
    - European Options 

Currently supports [DualNumbers.jl](https://github.com/JuliaDiff/DualNumbers.jl), [HyperDualNumbers.jl](https://github.com/JuliaDiff/HyperDualNumbers.jl)
for Automatic Differentiation (where it makes sense).

## How to Install
To install the package simply type on the Julia REPL the following:
```julia
] add FinancialFFT
```
## How to Test
After the installation, to test the package type on the Julia REPL the following:
```julia
] test FinancialFFT
```
## Hello World: Pricing European Call Option in Black Scholes Model
The following example shows how to price a european call option with underlying varying according to the Black Scholes Model, given the volatility:
```julia
#Import the Package
using FinancialFFT;

#Define Spot Datas
S0=100.0;
K=100.0;
r=0.02;
T=1.0;
d=0.01;
D=90.0;
#Define FinancialFFT Parameters
Nsim=10000;
Nstep=30;
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

## Curve Support
Non constant zero rates and dividend are managed.
An implied curve is built at time zero, such implied curve is able to return the right implied zero/dividend at a given time,
Without the need to carry the integral structure of the curve.
No support for multicurrency.