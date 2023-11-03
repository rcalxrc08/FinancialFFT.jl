using ChainRulesCore
"""
Struct for Lewis Integration Method

		bsProcess=LewisMethod(A,N)

Where:\n
		A	=	volatility of the process.
		N	=	volatility of the process.
"""
struct LewisMethod{num <: Number, num_1 <: Integer} <: AbstractIntegralMethod
    A::num
    N::num_1
    function LewisMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(A > 0.0, "A must be positive")
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        return new{num, num_1}(A, N)
    end
end
export LewisMethod;
#Equivalent to real(exp(z))
exp_mod(x) = exp(real_mod(x)) * cos(imag_mod(x))

using MuladdMacro
function evaluate_integrand_lewis_v(v, corr_adj, mcProcess::FinancialMonteCarlo.BaseProcess, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
    T = abstractPayoff.T
    v_im_adj = @. 1 // 2 + v * im
    # exp_mod = exp(corr_adj / 2)
    # real_res = @. FinancialFFT.real_mod(exp(corr_adj / 2 - adj_v * corr_adj + FinancialFFT.CharactheristicExponent_i(adj_v, mcProcess) * T)) / abs2(adj_v)
    real_res = @. FinancialFFT.exp_mod(v_im_adj * corr_adj + FinancialFFT.CharactheristicExponent_i(v_im_adj, mcProcess) * T) / abs2(v_im_adj)
    return real_res
end

# function evaluate_integrand_lewis_v(v, mod, corr, mcProcess::FinancialMonteCarlo.BaseProcess, abstractPayoff::FinancialMonteCarlo.BinaryEuropeanOption)
#     T = abstractPayoff.T
#     v_im = v * im
#     corr_adj = corr - mod
#     # return @. FinancialFFT.real_mod(exp(v_im * corr_adj + FinancialFFT.CharactheristicExponent_vi(1 // 2 - v_im, mcProcess) * T) / (1 // 2 - v_im))
#     return @. FinancialFFT.exp_mod(exp(v_im * corr_adj + FinancialFFT.CharactheristicExponent_vi(1 // 2 - v_im, mcProcess) * T) / (1 // 2 - v_im))
# end
function evaluate_integrand_lewis_v(v, corr_adj, mcProcess::FinancialMonteCarlo.BaseProcess, abstractPayoff::FinancialMonteCarlo.BinaryEuropeanOption)
    T = abstractPayoff.T
    v_im_adj = @. 1 // 2 + v * im
    return @. FinancialFFT.real_mod(exp(v_im_adj * corr_adj + FinancialFFT.CharactheristicExponent_i(v_im_adj, mcProcess) * T) * conj(v_im_adj)) / abs2(v_im_adj)
end
function convert_integral_result_to_call_price(sum_, S0_adj, K, mod, ::BinaryEuropeanOption)
    return S0_adj / K * exp(mod) / 2 * sum_
end

function convert_integral_result_to_call_price(sum_, S0_adj, _, mod, ::EuropeanOption)
    return S0_adj * (1 - exp(mod) / 2 * sum_)
end
using ChainRulesCore
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FinancialFFT.LewisMethod, abstractPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = abstractPayoff.T
    K = abstractPayoff.K
    A = method.A
    N = method.N
    S0 = mcProcess.underlying.S0
    corr = FinancialFFT.CharactheristicExponent_i(1, mcProcess) * T
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    S0_K = S0 / K
    mod = log(S0_K) + rT + dT
    eps_typed = ChainRulesCore.@ignore_derivatives eps(Float64)
    range_init = ChainRulesCore.@ignore_derivatives adapt_array(collect(range(-1, length = N, stop = 1)), mode)
    x_in = A * range_init
    x = eps_typed .+ x_in
    corr_adj = mod - corr
    y = evaluate_integrand_lewis_v(x, corr_adj, mcProcess, abstractPayoff)
    sum_ = sum(y)
    dx_adj = 2 * A / (N - 1)
    df = exp(-rT)
    ddf = exp(dT)
    S0_adj = S0 * ddf
    price = convert_integral_result_to_call_price(dx_adj * sum_ / π, S0_adj, K, -mod, abstractPayoff)
    return FinancialFFT.call_to_put(price, S0_adj, df, abstractPayoff)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoffs::Array{U}, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = abstractPayoffs[index_same_t]
        prices[index_same_t] .= [pricer(mcProcess, zero_rate, method, payoff) for payoff in payoffs]
    end

    return prices
end