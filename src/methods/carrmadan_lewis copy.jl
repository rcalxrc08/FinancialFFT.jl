"""
Struct for Lewis Integration Method

		bsProcess=CarrMadanLewisMethod(σ::num1) where {num1 <: Number}

Where:\n
		σ	=	volatility of the process.
"""
struct CarrMadanLewisMethod{num <: Number, num_1 <: Integer} <: AbstractFFTMethod
    A::num
    Npow::num_1
    function CarrMadanLewisMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        @assert(A > 0.0, "A must be positive")
        @assert(N > 2, "N must be greater than 2")
        @assert(N < 24, "N will cause overflow")
        return new{num, num_1}(A, N)
    end
end

export CarrMadanLewisMethod;
using DualMatrixTools
using Interpolations
"""
Documentation CarrMadanLewis Method
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanLewisMethod, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    A = method.A
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    correction = rT + dT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T

    idx = ChainRulesCore.@ignore_derivatives 0:(N-1)
    one_adj = ChainRulesCore.@ignore_derivatives Int8(1)
    minus_one_adj = ChainRulesCore.@ignore_derivatives Int8(-1)
    one_minus_one = ChainRulesCore.@ignore_derivatives @. ifelse(iseven(idx), one_adj, minus_one_adj)
    weights_ = ChainRulesCore.@ignore_derivatives @. (Int8(3) - one_minus_one)
    ChainRulesCore.@ignore_derivatives @views weights_[1] = one_adj
    ChainRulesCore.@ignore_derivatives @views weights_[end] = one_adj
    dx = A / N
    x = ChainRulesCore.@ignore_derivatives collect(idx * dx)
    x_im = @. 1 // 2 + x * im
    integrand = @. (one_minus_one * weights_) * exp(correction * x_im + FinancialFFT.characteristic_exponent_i(x_im, mcProcess) * T) / abs2(x_im)
    Y = fft(integrand)
    integral_value = @. real_mod(Y)
    pi_over_A = pi / A
    ks = pi_over_A * (-N .+ 2 * idx)
    # @show integral_value
    str_max = log(maximum(StrikeVec) / S0)
    str_min = log(minimum(StrikeVec) / S0)
    idx_max = findfirst(x -> x > 1.1 * str_max, ks)
    idx_min = findfirst(x -> x > 0.8 * str_min, ks)
    @views ks_adj = ks[idx_min:idx_max]
    @views integral_value_adj = integral_value[idx_min:idx_max]
    @show spline_cub = CubicSplineInterpolation(ks_adj, integral_value_adj) #wrong
    prices = @. S0 - sqrt(S0 * StrikeVec) * exp(-(rT + dT)) * dx / 3 * spline_cub(log(StrikeVec / S0)) / pi
    return prices * exp(dT)
end