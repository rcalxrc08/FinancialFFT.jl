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
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanLewisMethod) where {U <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    A = method.A
    d = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T) / T
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    cf = FinancialFFT.CharactheristicFunction(mcProcess, T)
    correction = (r - d) * T - FinancialFFT.CharactheristicExponent(-1im, mcProcess, T)
    CharFunc(v) = cf(v) * exp(v * 1im * correction)
    dx = A / N
    x = collect(0:(N-1)) * dx
    x[1] = 1e-312
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)
    @views weights_[1] = 1
    @views weights_[N] = 1

    dk = 2 * pi / A
    b = N * dk / 2
    ks = -b .+ dk * (0:(N-1))
    integrand = @. exp(-1im * b * x) * CharFunc(x - 0.5 * 1im) / (x^2 + 0.25) * weights_ * dx / 3
    fft!(integrand)
    integral_value = @. real_mod(integrand) / pi

    spline_cub = CubicSplineInterpolation(ks, integral_value)
    prices = @. S0 - sqrt(S0 * StrikeVec) * exp(-(r - d) * T) * spline_cub(log(StrikeVec / S0))
    return prices * exp(-d * T)
end