# Automatic differentiation

Prices are generic functions of the inputs of the model, the curve, and the payoff, so they can be differentiated by the
following back-ends, loaded with the corresponding package:

| Method | DualNumbers | HyperDualNumbers | TaylorSeries | ForwardDiff | ReverseDiff | Zygote | Mooncake |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `LewisMethod` | yes | yes | yes | yes | yes | yes | yes |
| `CosMethod` | yes | yes | yes | yes | yes | yes | yes |
| `CarrMadanLewisMethod` | yes | yes | yes | yes | yes | **no** | yes |
| `CarrMadanMethod` | yes | yes | yes | yes | yes | **no** | yes |
| `FFTDensityInverter` | yes | yes | yes | yes | yes | yes | yes |
| `CosDensityMethod` | yes | yes | yes | yes | yes | yes | yes |

The test suite checks each entry against the closed formulas of Black and Scholes, for European and binary options and
their smiles, with all of the inputs `S0`, `K`, `r`, `T`, `σ`, `d`. Gradients can be taken through the Heston model too.
TaylorSeries covers both `Taylor1` and `TaylorN`, which gives the mixed derivatives of several inputs at once:

```@example ad
using FinancialFFT, FinancialMonteCarlo, ForwardDiff

price(x) = pricer(BlackScholesProcess(x[3], Underlying(x[1], x[4])), ZeroRate(x[2]), LewisMethod(600.0, 20000), EuropeanOption(1.0, 100.0))
ForwardDiff.gradient(price, [100.0, 0.02, 0.2, 0.01])
```

`TaylorN` works in the same way; its variables are declared with `variables!`, or `set_variables` before TaylorSeries 0.23:

```julia
using TaylorSeries

δS, δσ = TaylorSeries.variables!("δS δσ"; order = 2)
pricer(BlackScholesProcess(0.2 + δσ, Underlying(100.0 + δS, 0.01)), ZeroRate(0.02), LewisMethod(600.0, 20000), EuropeanOption(1.0, 100.0))
```

Densities are differentiated in the same way, through `log(f(y))` (see [Densities](methods.md#Densities)):

| Density | DualNumbers | HyperDualNumbers | TaylorSeries | ForwardDiff | ReverseDiff | Zygote | Mooncake |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `FFTDensityInverter` | yes | yes | yes | yes | yes | yes | yes |
| `CosDensityMethod` | yes | yes | yes | yes | yes | yes | yes |

## Known limitations

!!! warning "Zygote with the FFT methods"
    The cubic spline of `CarrMadanMethod` and `CarrMadanLewisMethod` is built by mutating arrays, so it is excluded from
    the differentiation. Zygote does not raise an error, but the sensitivities that flow through the FFT are lost: the
    gradient in `σ` is zero, and those in `r`, `T` and `d` are wrong. Use another back-end with these methods.

- ReverseDiff.jl requires version 1.18 or later.
- On the GPU only ForwardDiff.jl and Zygote.jl are supported, see [GPU](@ref).
