# API reference

## Methods

```@docs
LewisMethod
CosMethod
CarrMadanLewisMethod
CarrMadanMethod
FinancialFFT.DensityInverter
```

## Pricing

```@docs
pricer(::FinancialMonteCarlo.BaseProcess, ::FinancialMonteCarlo.AbstractZeroRateCurve, ::LewisMethod, ::Any, ::FinancialMonteCarlo.BaseMode)
pricer(::FinancialMonteCarlo.BaseProcess, ::Array{U, 1}, ::FinancialMonteCarlo.AbstractZeroRateCurve, ::Number, ::CarrMadanMethod, ::FinancialMonteCarlo.BaseMode) where {U <: Number}
```

## Index

```@index
```
