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

function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::FinancialMonteCarlo.EuropeanOption)
    corr_h = corr_adj / 2
    corr_im = corr_adj * im
    return @. FinancialFFT.exp_mod(corr_h + v * corr_im + char_exp_v) / (1 // 4 + v^2)
end

function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::FinancialMonteCarlo.BinaryEuropeanOption)
    corr_h = corr_adj / 2
    corr_im = corr_adj * im
    return @. FinancialFFT.real_mod(exp(corr_h + v * corr_im + char_exp_v) / (1 // 2 + im * v))
end

function convert_integral_result_to_price(discounted_sum_, _, _, df, opt::BinaryEuropeanOption)
    C = df * discounted_sum_
    iscall = ChainRulesCore.@ignore_derivatives ifelse(opt.isCall, 1, 0)
    return iscall * C + (1 - iscall) * (df - C)
end

function convert_integral_result_to_price(discounted_sum_, S0, dT, df, opt::EuropeanOption)
    #simplify the following
    K_adj = opt.K * df
    C = S0 * exp(dT) - K_adj * discounted_sum_
    P = K_adj * (1 - discounted_sum_)
    return ifelse(opt.isCall, C, P)
end

struct LewisMethodResult{num_1, num_2}
    char_exp_v::num_1
    v_im_adj::num_2
    function LewisMethodResult(a::num_1, b::num_2) where {num_1, num_2}
        return new{num_1, num_2}(a, b)
    end
end

Base.broadcastable(x::LewisMethodResult) = Ref(x)

function lw_integrate(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
    corr_adj = log(S0 / abstractPayoff.K) + rT_corr + dT
    char_exp_v = char_exp_v_r.char_exp_v
    v_im_adj = char_exp_v_r.v_im_adj
    total_sum_lewis = sum(evaluate_integrand_lewis_v2(v_im_adj, corr_adj, char_exp_v, abstractPayoff))
    discounted_sum = dx_adj * total_sum_lewis / π
    return convert_integral_result_to_price(discounted_sum, S0, dT, df, abstractPayoff)
end

function lw_integrate_v(char_exp_v_r, abstractPayoff::Array, df, dT, dx_adj, S0, rT_corr)
    return @. lw_integrate(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
end

function lw_integrate_v(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
    return lw_integrate(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
end

function lw_pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FinancialFFT.LewisMethod, T, abstractPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    A = method.A
    N = method.N
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    eps_typed = ChainRulesCore.@ignore_derivatives eps(zero(Float64))
    range_init = ChainRulesCore.@ignore_derivatives adapt_array(collect(range(-1, length = N, stop = 1)), mode)
    one_half = 1 // 2
    v = ChainRulesCore.@ignore_derivatives @. (eps_typed + A * range_init)
    # v_im_adj = @. one_half + v * im
    char_exp_v = @. FinancialFFT.characteristic_exponent_i(one_half + v * im, mcProcess) * T
    lw_result = LewisMethodResult(char_exp_v, v)
    dx_adj = A / (N - 1)
    df = exp(-rT)
    rT_corr = rT - corr
    prices = lw_integrate_v(lw_result, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
    return prices
end

"""
Documentation LewisMethod Method
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FinancialFFT.LewisMethod, abstractPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    return lw_pricer(mcProcess, zero_rate, method, abstractPayoff.T, abstractPayoff, mode)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoffs::Array{U}, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = abstractPayoffs[index_same_t]
        prices[index_same_t] .= lw_pricer(mcProcess, zero_rate, method, T, payoffs, mode)
    end

    return prices
end