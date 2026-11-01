"""
    CosMethod(N)

Fourier-cosine expansion method of Fang & Oosterlee (2008). The density of the log-price is expanded on
a truncation interval `[a, b]` with the cosine terms `k = 0, …, N`; the interval is computed from a
Chernoff bound on the tails of the process (see `compute_chernoff_extrema_bisection`), so no range parameter is needed.
The expansion is on the put, and calls follow from parity: a call needs no information on the right tail of the process.

# Arguments
- `N`: number of cosine terms (`N > 2`).
"""
struct CosMethod{num_1 <: Integer} <: FinancialFFT.AbstractIntegralMethod
    N::num_1
    function CosMethod(N::num_1) where {num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        return new{num_1}(N)
    end
end
export CosMethod
# The expansion is on the put, whose payoff is bounded. The coefficients of a call grow like exp(b): they lose accuracy
# to rounding when the range is wide, and the heavy right tail they need is beyond it when it is not. Parity needs neither.
# Payoff of the put against cos(u (y - a)) for y = log(S_T / K) in [a, h], with k = 0 halved: adj flags that term.
function cos_binary_put_integral(u, adj, a, h)
    return (1 + !adj) * (adj * (h - a) + sin(u * (h - a)) / (adj + u))
end

function cos_put_integral(u, adj, a, h)
    sin_u, cos_u = sincos(u * (h - a))
    return cos_binary_put_integral(u, adj, a, h) - (1 + !adj) * (exp(h) * (cos_u + u * sin_u) - exp(a)) / (1 + u^2)
end

cos_payoff_integral(::VanillaPayoff) = cos_put_integral
cos_payoff_integral(::BinaryPayoff) = cos_binary_put_integral

put_to_option(P, S0_adj, df, opt::VanillaPayoff) = @. ifelse(opt.isCall, P + S0_adj - opt.K * df, P)
put_to_option(P, _, df, opt::BinaryPayoff) = @. ifelse(opt.isCall, df - P, P)

cos_series(terms::AbstractVector) = sum(terms)
cos_series(terms::AbstractMatrix) = vec(sum(terms; dims = 1))

function compute_chernorff_limits(Model::FinancialMonteCarlo.BaseProcess, rT, dT, T)
    drift = rT - dT
    # the tail is weighted by the scores in the derivatives: the error of the second one in σ is about 2500 ε
    epss = 1e-14
    s_min = 1.0
    s_max = 50.0
    x_toll_bisection = 1e-14
    compute_chernoff_extrema_bisection(Model, T, epss, drift, s_min, s_max, x_toll_bisection)
end
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
    N = method.N
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    K = adapt_array(opt.K, mode)
    # The limits are for log(S_T / S0): on y = log(S_T / K) they move with the strike, and the range covers all of them.
    # Sharing it among the strikes shares the payoff coefficients: only the phase below depends on the strike.
    a, b = ChainRulesCore.@ignore_derivatives begin
        a_S0, b_S0 = v_value_mod.(compute_chernorff_limits(mcProcess, rT, dT, T))
        K_min, K_max = extrema(K)
        v_value_mod(a_S0 + log(S0 / K_max)), v_value_mod(b_S0 + log(S0 / K_min))
    end
    bma = b - a
    u = ChainRulesCore.@ignore_derivatives FinancialFFT.adapt_array(collect((0:N) * (pi / bma)), mode)
    payoff = ChainRulesCore.@ignore_derivatives begin
        adj = PatternVector(length(u), InitialValuePattern(true, false))
        cos_payoff_integral(opt).(u, adj, a, max(a, min(0, b)))
    end
    driftT = rT - FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    v_char_exp = @. FinancialFFT.characteristic_exponent_i(u * im, mcProcess, T)
    coefficient = @. payoff * exp(FinancialFFT.real_mod(v_char_exp)) / bma
    phase = @. FinancialFFT.imag_mod(v_char_exp)
    S0_adj = S0 * exp(-dT)
    x = strike_columns(@. log(S0_adj / K) + driftT - a)
    df = exp(-rT)
    put_price = adjust_price_adimensional(cos_series(@. coefficient * cos(u * x + phase)), opt) .* df
    return put_to_option(put_price, S0_adj, df, opt)
end
