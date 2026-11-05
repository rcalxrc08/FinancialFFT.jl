# Shared by the densities and by the methods that price with them.

# Zero outside the range [f.a, f.b] of the density, where its expansion or its grid says nothing (the cosine series is
# periodic there). The test is on the values: the range is a constant for differentiation.
mask_outside_range(f, y::Number, value) = ifelse(f.a <= v_value_mod(y) <= f.b, value, zero(value))
function mask_outside_range(f, y::AbstractArray, value)
    is_inside = ChainRulesCore.@ignore_derivatives @. f.a <= v_value_mod(y) <= f.b
    return @. ifelse(is_inside, value, zero(value))
end

# Put payoff at a log-moneyness z <= 0, per unit of strike for vanilla options.
density_put_payoff(::VanillaPayoff) = z -> 1 - exp(z)
density_put_payoff(::BinaryPayoff) = z -> one(z)

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractDensityMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    f = density(mcProcess, r, method, opt.T, mode)
    return density_integral_price(mcProcess, r, f, method.M, opt, mode)
end

# Price of opt from a density f of the log-return: Simpson's rule on M intervals of log-moneyness for the put, whose
# integration domain ends at the strike, so that the kink of the payoff is an endpoint; the call by parity.
function density_integral_price(mcProcess, r, f, M, opt, mode)
    T = opt.T
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(r.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    K = adapt_array(opt.K, mode)
    log_K_S0 = strike_columns(@. log(K / S0))
    # log-moneyness z = log(S_T / K) from the left end of the range to the strike, one strike per column; the grid is
    # a constant for differentiation (Mooncake ignores @ignore_derivatives, hence the hook), and the density is ~ε at
    # its left end
    z, payoff = ChainRulesCore.@ignore_derivatives begin
        log_K_S0_val = ignore_derivatives_mod.(log_K_S0)
        z_min = @. f.a - log_K_S0_val
        z_max = @. max(z_min, min(0, f.b - log_K_S0_val))
        t = adapt_array(collect((0:M) ./ convert(typeof(f.a), M)), mode)
        z_grid = @. z_min + (z_max - z_min) * t
        simpson = PatternVector(M + 1, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
        put_payoff = density_put_payoff(opt)
        z_grid, @. simpson * put_payoff(z_grid) * (z_max - z_min) / (3 * M)
    end
    # the nodes are inside the range: no mask
    f_z = density_array(f, z .+ log_K_S0, mode)
    df = exp(-rT)
    put_price = adjust_price_adimensional(column_sums(payoff .* f_z), opt) .* df
    return put_to_option(put_price, S0 * exp(-dT), df, opt)
end
