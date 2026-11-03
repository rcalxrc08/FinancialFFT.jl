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
struct CosDensityMethod{num_1 <: Integer, num_2 <: Integer} <: AbstractIntegralMethod
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
# point instead of one cos per term.
function (f::CosDensity)(y)
    value = clenshaw_cos_sum(f.coefficients, (y - f.a) * (pi / (f.b - f.a)))
    # beyond the range the series is periodic, while the density is below ε
    is_inside = f.a <= v_value_mod(y) <= f.b
    return ifelse(is_inside, value, zero(value))
end

# The density at an array of points inside the range, with broadcasts and a reduction only: the coefficients are
# arguments of the broadcast, not captured, for ReverseDiff, and the kernels run on the GPU.
function cos_density_values(f::CosDensity, y::AbstractArray)
    y_vec = vec(y)
    values = sum(@. f.coefficients' * cos(f.u' * (y_vec - f.a)); dims = 2)
    return reshape(values, size(y))
end
cos_density_values(f::CosDensity, y::AbstractArray, ::FinancialMonteCarlo.BaseMode) = cos_density_values(f, y)
# On the CPU, the Clenshaw recurrence at each point (map, for ReverseDiff: the coefficients may be tracked).
function cos_density_values(f::CosDensity, y::AbstractArray, ::FinancialMonteCarlo.SerialMode)
    c = f.coefficients
    scale = pi / (f.b - f.a)
    return map(y_j -> clenshaw_cos_sum(c, (y_j - f.a) * scale), y)
end

# Put payoff at a log-moneyness z <= 0, per unit of strike for vanilla options.
density_put_payoff(::VanillaPayoff) = z -> 1 - exp(z)
density_put_payoff(::BinaryPayoff) = z -> one(z)

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosDensityMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
    M = method.M
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(r.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    f = density(mcProcess, r, method, T, mode)
    K = adapt_array(opt.K, mode)
    log_K_S0 = strike_columns(@. log(K / S0))
    # log-moneyness z = log(S_T / K) from the left end of the range to the strike, one strike per column; the grid is
    # a constant for differentiation, and the density is ~ε at its left end
    z, payoff = ChainRulesCore.@ignore_derivatives begin
        log_K_S0_val = v_value_mod.(log_K_S0)
        z_min = @. f.a - log_K_S0_val
        z_max = @. max(z_min, min(0, f.b - log_K_S0_val))
        t = adapt_array(collect((0:M) ./ convert(typeof(f.a), M)), mode)
        z_grid = @. z_min + (z_max - z_min) * t
        simpson = PatternVector(M + 1, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
        put_payoff = density_put_payoff(opt)
        z_grid, @. simpson * put_payoff(z_grid) * (z_max - z_min) / (3 * M)
    end
    f_z = cos_density_values(f, z .+ log_K_S0, mode)
    df = exp(-rT)
    put_price = adjust_price_adimensional(cos_series(payoff .* f_z), opt) .* df
    return put_to_option(put_price, S0 * exp(-dT), df, opt)
end
