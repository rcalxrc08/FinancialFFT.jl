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
function evaluate_integrand_lewis_v(v_im_adj, corr_adj, mcProcess::FinancialMonteCarlo.BaseProcess, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
    T = abstractPayoff.T
    return @. FinancialFFT.exp_mod(v_im_adj * corr_adj + FinancialFFT.characteristic_exponent_i(v_im_adj, mcProcess) * T) / abs2(v_im_adj)
end

function evaluate_integrand_lewis_v(v_im_adj, corr_adj, mcProcess::FinancialMonteCarlo.BaseProcess, abstractPayoff::FinancialMonteCarlo.BinaryEuropeanOption)
    T = abstractPayoff.T
    return @. FinancialFFT.real_mod(exp(v_im_adj * corr_adj + FinancialFFT.characteristic_exponent_i(v_im_adj, mcProcess) * T) * conj(v_im_adj)) / abs2(v_im_adj)
end

function convert_integral_result_to_price(discounted_sum_, _, _, df, opt::BinaryEuropeanOption)
    C = discounted_sum_
    iscall = ChainRulesCore.@ignore_derivatives ifelse(opt.isCall, 1, 0)
    return iscall * C + (1 - iscall) * (df - C)
end

function convert_integral_result_to_price(discounted_sum_, S0, dT, df, opt::EuropeanOption)
    C = S0 * exp(dT) - opt.K * discounted_sum_
    P = opt.K * (df - discounted_sum_)
    return ifelse(opt.isCall, C, P)
end

"""
Documentation LewisMethod Method
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FinancialFFT.LewisMethod, abstractPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = abstractPayoff.T
    K = abstractPayoff.K
    A = method.A
    N = method.N
    S0 = mcProcess.underlying.S0
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    S0_K = S0 / K
    mod = log(S0_K) + rT + dT
    eps_typed = ChainRulesCore.@ignore_derivatives eps(Float64)
    range_init = ChainRulesCore.@ignore_derivatives adapt_array(collect(range(-1, length = N, stop = 1)), mode)
    x_in = A * range_init
    x = eps_typed .+ x_in
    corr_adj = mod - corr
    v_im_adj = @. (1 + x * (2 * im)) / 2 # I can't use Irrationals because of https://github.com/JuliaGPU/CUDA.jl/issues/1926
    y = evaluate_integrand_lewis_v(v_im_adj, corr_adj, mcProcess, abstractPayoff)
    sum_ = sum(y)
    dx_adj = A / (N - 1)
    df = exp(-rT)
    discounted_sum = dx_adj * sum_ / π * df
    price = convert_integral_result_to_price(discounted_sum, S0, dT, df, abstractPayoff)
    return price
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