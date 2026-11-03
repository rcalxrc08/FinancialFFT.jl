using FinancialMonteCarlo
import FinancialMonteCarlo.pricer, FinancialMonteCarlo.AbstractMethod
abstract type AbstractIntegrationMethod <: FinancialMonteCarlo.AbstractMethod end
abstract type AbstractFFTMethod <: AbstractIntegrationMethod end
abstract type AbstractIntegralMethod <: AbstractIntegrationMethod end
Base.broadcastable(x::AbstractIntegrationMethod) = Ref(x)

"""
    AbstractDensity

Density of the log-return `log(S_T / S_0)` returned by [`density`](@ref): `d(y)` evaluates it at a log-return `y`, and
it is zero outside `[d.a, d.b]`.
"""
abstract type AbstractDensity end
Base.broadcastable(x::AbstractDensity) = Ref(x)

"""
    density(mcProcess, zero_rate, method, T[, mode])

Density of the log-return `log(S_T / S_0)` of `mcProcess` at the horizon `T`, under the risk-neutral measure of
`zero_rate`, computed with `method` (`CosDensityMethod` or `DensityInverter`). Returns an [`AbstractDensity`](@ref),
to be evaluated at any number of points: the setup runs once.

It does not depend on `S_0`: the density of `log(S_T)` given `S_t` is `d(x - log(S_t))`, with the horizon of the step
as `T` for a Lévy process. For Heston it is the marginal density, not the transition density given the variance. It
is generic in the number type of the parameters of the model, so scores such as `∂θ log f(y)` come from automatic
differentiation of `log(d(y))`; its range is a constant for differentiation. The evaluation of a point runs on the CPU.
"""
function density end
export density
