# The interface of the densities: density computes one, and every AbstractDensity is evaluated at a point and at an
# array of points in the same way. A density type implements, for its points inside [a, b]:
#   density_value(f, y)               the value at a number y; it must not fail outside [a, b], where it is masked
#   density_array(f, y[, mode])       the values at an array y, with its shape (default: density_value at each point);
#                                     the mode lets the pricers choose a CPU or a GPU kernel

"""
    AbstractDensity

Density of the log-return `log(S_T / S_0)` returned by [`density`](@ref): `f(y)` evaluates it at a log-return `y`, on
the CPU, and [`FinancialFFT.density_values`](@ref)`(f, y)` at an array of log-returns. It is zero outside
`[f.a, f.b]`.
"""
abstract type AbstractDensity end
Base.broadcastable(x::AbstractDensity) = Ref(x)

"""
    density(mcProcess, zero_rate, method, T)

Density of the log-return `log(S_T / S_0)` of `mcProcess` at the horizon `T`, under the risk-neutral measure of
`zero_rate`, computed with `method` (`CosDensityMethod` or `FFTDensityInverter`). Returns an [`AbstractDensity`](@ref),
to be evaluated at any number of points: the setup runs once.

It does not depend on `S_0`: the density of `log(S_T)` given `S_t` is `f(x - log(S_t))`, with the horizon of the step
as `T` for a Lévy process. For Heston it is the marginal density, not the transition density given the variance. It
is generic in the number type of the parameters of the model, so scores such as `∂θ log f(y)` come from automatic
differentiation of `log(f(y))`; its range is a constant for differentiation. The evaluation of a point runs on the CPU.

The method carries the execution mode: with `FinancialMonteCarlo.CudaMode()` (the `mode` keyword of its constructor,
or `FinancialMonteCarlo.with_mode(method, mode)`) the density is computed and stored on the GPU.
"""
function density end
export density

(f::AbstractDensity)(y) = mask_outside_range(f, y, density_value(f, y))

"""
    FinancialFFT.density_values(f, y)

Values of the density `f` (an [`FinancialFFT.AbstractDensity`](@ref)) at the array of log-returns `y`, with the shape
of `y`. For `FFTDensity` and `CosDensity` it uses broadcasts, gathers and reductions only: it runs on the device of
`y` and of the density, and Zygote differentiates it there too.
"""
density_values(f::AbstractDensity, y::AbstractArray) = mask_outside_range(f, y, density_array(f, y))

density_array(f::AbstractDensity, y::AbstractArray) = map(y_j -> density_value(f, y_j), y)
density_array(f::AbstractDensity, y::AbstractArray, ::FinancialMonteCarlo.AbstractMode) = density_array(f, y)

# Before FinancialMonteCarlo 0.6 the mode was the last argument of pricer and density: deprecated for one release.
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractIntegrationMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.AbstractMode)
    Base.depwarn("`pricer(model, zero_rate, method, payoff, mode)` is deprecated, the mode is part of the method: use `pricer(model, zero_rate, FinancialMonteCarlo.with_mode(method, mode), payoff)` instead.", :pricer)
    return pricer(mcProcess, zero_rate, FinancialMonteCarlo.with_mode(method, mode), opt)
end
function density(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractDensityMethod, T, mode::FinancialMonteCarlo.AbstractMode)
    Base.depwarn("`density(model, zero_rate, method, T, mode)` is deprecated, the mode is part of the method: use `density(model, zero_rate, FinancialMonteCarlo.with_mode(method, mode), T)` instead.", :density)
    return density(mcProcess, zero_rate, FinancialMonteCarlo.with_mode(method, mode), T)
end
