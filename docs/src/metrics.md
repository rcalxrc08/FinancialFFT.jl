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
pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanLewisMethod) where {U <: Number}
FinancialMonteCarlo.pricer::Union{Tuple{U}, Tuple{FinancialMonteCarlo.BaseProcess, Vector{U}, FinancialMonteCarlo.AbstractZeroRateCurve, Number, CarrMadanMethod}, Tuple{FinancialMonteCarlo.BaseProcess, Vector{U}, FinancialMonteCarlo.AbstractZeroRateCurve, Number, CarrMadanMethod, FinancialMonteCarlo.BaseMode}} where U<:Number
FinancialMonteCarlo.pricer::Union{Tuple{FinancialMonteCarlo.BaseProcess, FinancialMonteCarlo.AbstractZeroRateCurve, LewisMethod, Any}, Tuple{FinancialMonteCarlo.BaseProcess, FinancialMonteCarlo.AbstractZeroRateCurve, LewisMethod, Any, FinancialMonteCarlo.BaseMode}}
FinancialMonteCarlo.pricer::Union{Tuple{U}, Tuple{FinancialMonteCarlo.BaseProcess, FinancialMonteCarlo.AbstractZeroRateCurve, LewisMethod, Array{U}}, Tuple{FinancialMonteCarlo.BaseProcess, FinancialMonteCarlo.AbstractZeroRateCurve, LewisMethod, Array{U}, FinancialMonteCarlo.BaseMode}} where U<:EuropeanOption
FinancialMonteCarlo.pricer::Union{Tuple{U}, Tuple{FinancialMonteCarlo.BaseProcess, Vector{U}, FinancialMonteCarlo.AbstractZeroRateCurve, Number, CarrMadanLewisMethod}, Tuple{FinancialMonteCarlo.BaseProcess, Vector{U}, FinancialMonteCarlo.AbstractZeroRateCurve, Number, CarrMadanLewisMethod, FinancialMonteCarlo.BaseMode}} where U<:Number
```