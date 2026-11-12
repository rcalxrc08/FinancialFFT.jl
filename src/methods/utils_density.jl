# Shared by the densities and by the methods that price with them.

# Zero outside the range [f.a, f.b] of the density, where its expansion or its grid says nothing (the cosine series is
# periodic there). The test is on the values: the range is a constant for differentiation.
mask_outside_range(f, y::Number, value) = ifelse(f.a <= v_value_mod(y) <= f.b, value, zero(value))
# The test at an array of points, a constant for differentiation, in a function of its own that the Enzyme extension marks
# inactive (it cannot tell the ends of the range, fields of a density with active values, from active memory).
inside_range(a, b, y) = @. a <= v_value_mod(y) <= b
function mask_outside_range(f, y::AbstractArray, value)
    is_inside = ChainRulesCore.@ignore_derivatives inside_range(f.a, f.b, y)
    return @. ifelse(is_inside, value, zero(value))
end

# Put payoff at a log-moneyness z <= 0, per unit of strike for vanilla options.
density_put_payoff(::VanillaPayoff) = z -> 1 - exp(z)
density_put_payoff(::BinaryPayoff) = z -> one(z)

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractDensityMethod, opt_::FourierPayoff)
    mode = FinancialMonteCarlo.execution_mode(method)
    opt = FinancialMonteCarlo.to_mode(opt_, mode)
    f = density(mcProcess, r, method, opt.T)
    return density_integral_price(mcProcess, r, f, method.M, opt, mode)
end

# Log-moneyness z = log(S_T / K) from the left end a of the range of a density to the strike, M Simpson intervals for each
# strike (one per column), and the weights of Simpson's rule times the put payoff: a constant for differentiation, in a
# function of its own so that the Enzyme extension can mark it inactive (Enzyme ignores @ignore_derivatives, and cannot
# tell that the ends of the range, fields of a density with active values, are constants). The density is ~ε at a.
function density_integration_grid(a, b, log_K_S0_val, M, opt, mode)
    z_min = @. a - log_K_S0_val
    z_max = @. max(z_min, min(0, b - log_K_S0_val))
    # a vector, not a float range (see the nodes of carrmadan.jl): k / M is rounded once, as the range is
    M_FT = convert(typeof(a), M)
    t = adapt_array([k / M_FT for k = 0:M], mode)
    z_grid = @. z_min + (z_max - z_min) * t
    simpson = PatternVector(M + 1, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
    put_payoff = density_put_payoff(opt)
    return z_grid, @. simpson * put_payoff(z_grid) * (z_max - z_min) / (3 * M)
end

# Price of opt from a density f of the log-return: Simpson's rule on M intervals of log-moneyness for the put, whose
# integration domain ends at the strike, so that the kink of the payoff is an endpoint; the call by parity.
function density_integral_price(mcProcess, r, f, M, opt, mode)
    T = opt.T
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(r.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    K = opt.K
    # one strike per column
    log_K_S0 = transpose(@. log(K / S0))
    # the grid is a constant for differentiation (Mooncake ignores @ignore_derivatives, hence the hook)
    z, payoff = ChainRulesCore.@ignore_derivatives density_integration_grid(f.a, f.b, ignore_derivatives_mod(log_K_S0), M, opt, mode)
    # the nodes are inside the range: no mask
    f_z = density_array(f, z .+ log_K_S0, mode)
    df = exp(-rT)
    put_price = adjust_price_adimensional(column_sums(payoff .* f_z), opt) .* df
    return put_to_option(put_price, S0 * exp(-dT), df, opt)
end
