# [Metrics](@id Metric)

The following type of *Metric* are supported from the package:

* `pricer`
* `delta`
* `rho`
* `variance`
* `confinter`

## Common Interface

Each Metric must implement its own *Metric* method; the general interface is the following:
```@docs
pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::FinancialFFT.CarrMadanMethod) where {U <: Number}
pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanLewisMethod) where {U <: Number}
```