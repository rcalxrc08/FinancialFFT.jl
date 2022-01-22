# FinancialFFT.jl

FinancialFFT.jl contains the following capabilities:

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

## Index

```@index
```