"""
    CosDensityMethod(N = 128, M = 2^10)

Density of the log-return expanded on `N + 1` cosine terms (Fang & Oosterlee, 2008) on the range of `CosMethod`, see
[`density`](@ref). As a pricer, it integrates the payoff against that density by Simpson's rule on `M` intervals of
log-moneyness. Like `CosMethod`, it prices the put, whose integration domain ends at the strike, so that the kink of
the payoff is an endpoint, and recovers the call by parity. Supports `EuropeanOption`, `BinaryEuropeanOption` and
their smiles.

# Arguments
- `N`: number of cosine terms of the density (`N > 2`).
- `M`: number of Simpson intervals for each strike (even, `M > 2`); the density does not use it.
"""
struct CosDensityMethod{num_1 <: Integer, num_2 <: Integer} <: AbstractDensityMethod
    N::num_1
    M::num_2
    function CosDensityMethod(N::num_1 = 128, M::num_2 = 2^10) where {num_1 <: Integer, num_2 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        ChainRulesCore.@ignore_derivatives @assert(M > 2 && iseven(M), "M must be even and greater than 2")
        return new{num_1, num_2}(N, M)
    end
end
export CosDensityMethod

"""
    CosDensity

Density of the log-return of [`CosDensityMethod`](@ref), returned by [`density`](@ref). The range `[a, b]` is the one
of `CosMethod`, from a Chernoff bound on the tails with `ε = 1e-14`. The expansion converges exponentially for smooth
densities: `N = 64` already reaches machine precision in Black Scholes. The error is absolute: far in the tails, where
the density is below about `1e-14`, it can be slightly negative, and its relative error, which is the one of a score,
grows.
"""
struct CosDensity{num_1, num_2, num_3} <: AbstractDensity
    coefficients::num_1
    u::num_2
    a::num_3
    b::num_3
    function CosDensity(coefficients::num_1, u::num_2, a::num_3, b::num_3) where {num_1, num_2, num_3}
        return new{num_1, num_2, num_3}(coefficients, u, a, b)
    end
end

function density(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosDensityMethod, T, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    N = method.N
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    a, b = ChainRulesCore.@ignore_derivatives v_value_mod.(compute_chernorff_limits(mcProcess, rT, dT, T))
    bma = b - a
    u = ChainRulesCore.@ignore_derivatives adapt_array(collect((0:N) * (pi / bma)), mode)
    # the k = 0 term of the cosine series is halved
    weight = ChainRulesCore.@ignore_derivatives PatternVector(N + 1, InitialValuePattern(Int8(1), Int8(2)))
    drift_minus_a = rT - dT - FinancialFFT.characteristic_exponent_i(1, mcProcess, T) - a
    # Re[φ(u) exp(-i u a)] for the characteristic function φ of log(S_T / S_0)
    coefficients = @. weight * exp_mod(FinancialFFT.characteristic_exponent_i(u * im, mcProcess, T) + u * drift_minus_a * im) / bma
    return CosDensity(coefficients, u, a, b)
end

# Clenshaw recurrence of Σ c_k cos(k θ), in Reinsch's form (the ends of the range are θ = 0 and π): one sincos per
# point instead of one cos per term. Beyond the range the series is periodic: the interface masks it.
density_value(f::CosDensity, y) = clenshaw_cos_sum(f.coefficients, (y - f.a) * (pi / (f.b - f.a)))

# At an array of points, with broadcasts and a reduction only: the coefficients are arguments of the broadcast, not
# captured, for ReverseDiff, and the kernels run on the GPU.
function density_array(f::CosDensity, y::AbstractArray)
    y_vec = vec(y)
    values = sum(@. f.coefficients' * cos(f.u' * (y_vec - f.a)); dims = 2)
    return reshape(values, size(y))
end
# On the CPU, the Clenshaw recurrence at each point (map, for ReverseDiff: the coefficients may be tracked).
density_array(f::CosDensity, y::AbstractArray, ::FinancialMonteCarlo.SerialMode) = map(y_j -> density_value(f, y_j), y)
