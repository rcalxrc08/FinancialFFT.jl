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

# Σ_k coefficient_k cos(u_k x + phase_k) for each x (a number or a vector, one per strike).
# On the CPU it is the real part of Σ_k C_k exp(i k θ), with C_k = coefficient_k exp(i phase_k) and θ = π x / (b - a): the
# Clenshaw recurrence costs a sincos per strike instead of a cos per term and strike, and the phases (which
# do not depend on the strike) are folded into C_k once. Other modes (GPU) cannot index the coefficients, and keep one cos per term.
function cos_phased_series(coefficient, phase, u, x, ::FinancialMonteCarlo.SerialMode)
    sin_cos = sincos_mod.(phase)
    c_re = @. coefficient * last(sin_cos)
    c_im = @. coefficient * first(sin_cos)
    θ = x .* (last(u) / (length(u) - 1))
    return map(θ_j -> cos_clenshaw(c_re, c_im, θ_j), θ)
end
function cos_phased_series(coefficient, phase, u, x, ::FinancialMonteCarlo.BaseMode)
    x_columns = strike_columns(x)
    return column_sums(@. coefficient * cos(u * x_columns + phase))
end

# Re Σ_k (c_re_k + i c_im_k) exp(i k θ) = Σ_k c_re_k cos(k θ) - c_im_k sin(k θ).
cos_clenshaw(c_re, c_im, θ) = clenshaw_exp_sum(c_re, c_im, θ, Val(true))

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
    x = @. log(S0_adj / K) + driftT - a
    df = exp(-rT)
    put_price = adjust_price_adimensional(cos_phased_series(coefficient, phase, u, x, mode), opt) .* df
    return put_to_option(put_price, S0_adj, df, opt)
end
