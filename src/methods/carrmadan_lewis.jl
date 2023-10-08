"""
Struct for Lewis Integration Method

		bsProcess=CarrMadanLewisMethod(σ::num1) where {num1 <: Number}

Where:\n
		σ	=	volatility of the process.
"""
mutable struct CarrMadanLewisMethod{num <: Number, num_1 <: Integer} <: AbstractFFTMethod
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
    correction = rT + dT - FinancialFFT.CharactheristicExponent(-1im, mcProcess, T)
    dx = A / N
    x = collect(0:(N-1)) * dx
    ChainRulesCore.@ignore_derivatives x[1] = eps(zero(Float64)) #wrong
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)
    ChainRulesCore.@ignore_derivatives @views weights_[1] = 1
    ChainRulesCore.@ignore_derivatives @views weights_[N] = 1
    dk = 2 / A * pi
    b = N * dk / 2
    integrand = @. exp(T * FinancialFFT.CharactheristicExponent(x - im // 2, mcProcess) + x * im * (correction - b) + correction / 2) / (x^2 + 1 // 4) * weights_ * dx / 3
    fft!(integrand)
    integral_value = @. real_mod(integrand) / pi

    ks = -b .+ dk * (0:(N-1))
    spline_cub = ChainRulesCore.@ignore_derivatives CubicSplineInterpolation(ks, integral_value) #wrong
    prices = @. S0 - sqrt(S0 * StrikeVec) * exp(-(rT + dT)) * spline_cub(log(StrikeVec / S0))
    return prices * exp(dT)
end