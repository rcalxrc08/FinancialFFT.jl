"""
    CosDensity

Density of the log-return `log(S_T / S_0)` recovered by a cosine expansion (Fang & Oosterlee, 2008) on the range of
`CosMethod`. Built by [`cos_density`](@ref); `d(y)` evaluates it at a log-return `y`, and is zero outside `[d.a, d.b]`.
It does not depend on `S_0`: the density of `log(S_T)` given `S_t` is `d(x - log(S_t))` for a Lévy process, with the
horizon of the step as `T`.
"""
struct CosDensity{num_1, num_2, num_3}
    coefficients::num_1
    u::num_2
    a::num_3
    b::num_3
    function CosDensity(coefficients::num_1, u::num_2, a::num_3, b::num_3) where {num_1, num_2, num_3}
        return new{num_1, num_2, num_3}(coefficients, u, a, b)
    end
end

"""
    cos_density(mcProcess, zero_rate, T, N = 128, mode = SerialMode())

Density of the log-return `log(S_T / S_0)` of `mcProcess` under the risk-neutral measure of `zero_rate`, expanded on
`N + 1` cosine terms. The range `[a, b]` is the one of `CosMethod`, from a Chernoff bound on the tails with
`ε = 1e-14`. Returns a [`CosDensity`](@ref), to be evaluated at any number of points: the setup, which computes the
range and the characteristic function, runs once.

The result is generic in the number type of the parameters of the model, so scores such as `∂θ log f(y)` come from
automatic differentiation of `log(d(y))`. The range is a constant for differentiation. The expansion converges
exponentially for smooth densities: `N = 64` already reaches machine precision in Black Scholes. The error is absolute:
far in the tails, where the density is below about `1e-14`, it can be slightly negative, and its relative error, which
is the one of a score, grows.
"""
function cos_density(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T, N::Integer = 128, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
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

# Clenshaw recurrence of Σ c_k cos(k θ) = Σ c_k T_k(cos θ): one cos per point instead of one per term.
function (density::CosDensity)(y)
    c = density.coefficients
    t = cos((y - density.a) * (pi / (density.b - density.a)))
    b_1 = b_2 = zero(t * first(c))
    for k in lastindex(c):-1:(firstindex(c)+1)
        b_1, b_2 = @muladd(c[k] + 2 * t * b_1 - b_2), b_1
    end
    value = first(c) + t * b_1 - b_2
    # beyond the range the series is periodic, while the density is below ε
    is_inside = density.a <= v_value_mod(y) <= density.b
    return ifelse(is_inside, value, zero(value))
end

# The density at an array of points inside the range, with broadcasts and a reduction only: the coefficients are
# arguments of the broadcast, not captured, for ReverseDiff, and the kernels run on the GPU.
function cos_density_values(density::CosDensity, y::AbstractArray)
    y_vec = vec(y)
    values = sum(@. density.coefficients' * cos(density.u' * (y_vec - density.a)); dims = 2)
    return reshape(values, size(y))
end

"""
    CosDensityMethod(N = 128, M = 2^10)

Integrates the payoff against the density of [`cos_density`](@ref), with `N + 1` cosine terms, by Simpson's rule on
`M` intervals of log-moneyness. Like `CosMethod`, it prices the put, whose integration domain ends at the strike, so
that the kink of the payoff is an endpoint, and recovers the call by parity. Supports `EuropeanOption`,
`BinaryEuropeanOption` and their smiles.

# Arguments
- `N`: number of cosine terms of the density (`N > 2`).
- `M`: number of Simpson intervals for each strike (even, `M > 2`).
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
export CosDensityMethod, cos_density

# Put payoff at a log-moneyness z <= 0, per unit of strike for vanilla options.
density_put_payoff(::VanillaPayoff) = z -> 1 - exp(z)
density_put_payoff(::BinaryPayoff) = z -> one(z)

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosDensityMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
    M = method.M
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(r.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    density = cos_density(mcProcess, r, T, method.N, mode)
    K = adapt_array(opt.K, mode)
    log_K_S0 = strike_columns(@. log(K / S0))
    # log-moneyness z = log(S_T / K) from the left end of the range to the strike, one strike per column; the grid is
    # a constant for differentiation, and the density is ~ε at its left end
    z, payoff = ChainRulesCore.@ignore_derivatives begin
        log_K_S0_val = v_value_mod.(log_K_S0)
        z_min = @. density.a - log_K_S0_val
        z_max = @. max(z_min, min(0, density.b - log_K_S0_val))
        t = adapt_array(collect((0:M) / M), mode)
        z_grid = @. z_min + (z_max - z_min) * t
        simpson = PatternVector(M + 1, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
        put_payoff = density_put_payoff(opt)
        z_grid, @. simpson * put_payoff(z_grid) * (z_max - z_min) / (3 * M)
    end
    f = cos_density_values(density, z .+ log_K_S0)
    df = exp(-rT)
    put_price = adjust_price_adimensional(cos_series(payoff .* f), opt) .* df
    return put_to_option(put_price, S0 * exp(-dT), df, opt)
end
