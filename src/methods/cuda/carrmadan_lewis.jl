#TODO: CHANGE NAME
function pricer_cu(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::num, method::CarrMadanLewisMethod) where {U <: Number, num <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    A = method.A
    d = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T) / T
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    cf = FinancialFFT.CharactheristicFunction(mcProcess, T)
    corr = (r - d) * T - FinancialFFT.CharactheristicExponent(-1im, mcProcess, T)
    CharFunc(v::num) where {num <: Number} = cf(v) * exp(v * 1im * corr)
    dx = A / N
    x = collect(0:(N-1)) * dx
    x[1] = 1e-312
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)
    @views weights_[1] = 1
    @views weights_[N] = 1
    weights_cu = cu(weights_)
    x_cu = cu(x)
    dk = 2 * pi / A
    b = N * dk / 2
    ks = -b .+ dk * (0:(N-1))
    integrand = @. exp(-1im * b * x_cu) * CharFunc(x_cu - 0.5 * 1im) / (x_cu^2 + 0.25) * weights_cu * dx / 3
    fft!(integrand)
    integral_value = collect(@. real_mod(integrand) / pi)

    spline_cub = CubicSplineInterpolation(ks, integral_value)
    prices = @. S0 - sqrt(S0 * StrikeVec) * exp(-(r - d) * T) * spline_cub(log(StrikeVec / S0))
    return prices * exp(-d * T)
end