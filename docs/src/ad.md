# Automatic differentiation

Prices are generic functions of the inputs of the model, the curve, and the payoff, so they can be differentiated by the
following back-ends, loaded with the corresponding package:

| Method | DualNumbers | HyperDualNumbers | TaylorSeries | ForwardDiff | ReverseDiff | Zygote | Mooncake |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `LewisMethod` | yes | yes | yes | yes | yes | yes | yes |
| `CosMethod` | yes | yes | yes | yes | yes | yes | yes |
| `CarrMadanLewisMethod` | yes | yes | yes | yes | yes | **no** | yes |
| `CarrMadanMethod` | yes | yes | yes | yes | yes | **no** | yes |
| `DensityInverter` | yes | yes | **no** | yes | yes | yes | yes |

The test suite checks each entry against the closed formulas of Black and Scholes, for European and binary options and
their smiles, with all of the inputs `S0`, `K`, `r`, `T`, `σ`, `d`. Gradients can be taken through the Heston model too.

```@example ad
using FinancialFFT, FinancialMonteCarlo, ForwardDiff

price(x) = pricer(BlackScholesProcess(x[3], Underlying(x[1], x[4])), ZeroRate(x[2]), LewisMethod(600.0, 20000), EuropeanOption(1.0, 100.0))
ForwardDiff.gradient(price, [100.0, 0.02, 0.2, 0.01])
```

## Known limitations

!!! warning "Zygote with the FFT methods"
    The cubic spline of `CarrMadanMethod` and `CarrMadanLewisMethod` is built by mutating arrays, so it is excluded from
    the differentiation. Zygote does not raise an error, but the sensitivities that flow through the FFT are lost: the
    gradient in `σ` is zero, and those in `r`, `T` and `d` are wrong. Use another back-end with these methods.

- `TaylorSeries` results of `DensityInverter` do not match the closed formulas.
- ReverseDiff.jl requires version 1.18 or later.
- On the GPU only ForwardDiff.jl and Zygote.jl are supported, see [GPU](@ref).
