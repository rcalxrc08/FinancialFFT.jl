function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanMethod, ::FinancialMonteCarlo.CudaMode) where {U <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    d = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T) / T
    A = method.A
    #* v-> compute integral as a summation
    dx = A / N
    real_vec = 0:(N-1)
    v = collect(real_vec) * dx            # the final value A is excluded
    v[1] = 1e-312
    corr = FinancialFFT.CharactheristicExponent(-1im, mcProcess, T)
    CharFunc(v) = exp(CharactheristicExponent(v, mcProcess, T) - v * 1im * corr)
    integrand_f(v) = exp(1im * (r - d) * v * T) * (CharFunc(v - 1im) - 1) / (1im * v - v^2)
    # Option Price
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)# Simpson weights
    @views weights_[1] = 1
    @views weights_[end] = 1
    dk = 2 * pi / A
    b = N * dk / 2
    weights_cu = cu(weights_)
    v_cu = cu(v)
    complex_vec_z_k = @. integrand_f(v_cu) * dx * weights_cu / 3 * exp(1im * b * v_cu)
    fft!(complex_vec_z_k)
    z_T = collect(@. real_mod(complex_vec_z_k) / pi)

    ks = -b .+ dk * real_vec
    spline_cub = CubicSplineInterpolation(ks, z_T)
    k_interp = @. log(StrikeVec / S0)
    C = @. S0 * spline_cub(k_interp) + max(S0 - StrikeVec * exp(-(r - d) * T), 0)
    return C * exp(-d * T)
end