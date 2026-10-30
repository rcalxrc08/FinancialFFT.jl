"""
    CarrMadanLewisMethod(A, Npow)

FFT evaluation of the Lewis (2001) formula: the integrand along the line `Im(u) = 1/2` is sampled
on `[0, A)` with `2^Npow` nodes and Simpson weights, a single FFT returns the integral on a log-strike grid,
and prices at the requested strikes are obtained through cubic-spline interpolation.

# Arguments
- `A`: upper bound of the frequency integration domain (`A > 0`).
- `Npow`: base-2 logarithm of the number of FFT nodes (`2 < Npow < 24`).

The spline step is excluded from ChainRules-based differentiation, so Zygote gradients are not reliable for this method.
"""
struct CarrMadanLewisMethod{num <: Number, num_1 <: Integer} <: AbstractFFTMethod
    A::num
    Npow::num_1
    function CarrMadanLewisMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(A > 0.0, "A must be positive")
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        ChainRulesCore.@ignore_derivatives @assert(N < 24, "N will cause overflow")
        return new{num, num_1}(A, N)
    end
end

export CarrMadanLewisMethod;
using Interpolations
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanLewisMethod, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: Number}
    Npow = method.Npow
    N = ChainRulesCore.@ignore_derivatives 2^Npow
    S0 = mcProcess.underlying.S0
    A = method.A
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    correction = rT + dT - FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    idx = ChainRulesCore.@ignore_derivatives 0:(N-1)
    # one_minus_one = ChainRulesCore.@ignore_derivatives AlternateVector(Int8(1), Int8(-1), N)
    one_minus_one = ChainRulesCore.@ignore_derivatives PatternVector(N, EvenOddPattern(Int8(1), Int8(-1)))
    # weights_simpson = ChainRulesCore.@ignore_derivatives AlternatePaddedVector(Int8(1), Int8(4), Int8(2), Int8(1), N)
    weights_simpson = ChainRulesCore.@ignore_derivatives PatternVector(N, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
    support_array = ChainRulesCore.@ignore_derivatives @. one_minus_one * weights_simpson
    dx = A / N
    x = ChainRulesCore.@ignore_derivatives adapt_array(collect(idx * dx), mode)
    x_im = @. 1 // 2 + x * im
    Y = fft(@. support_array * exp(correction * x_im + FinancialFFT.characteristic_exponent_i(x_im, mcProcess, T)) / abs2(x_im))
    integral_value = @. real_mod(Y)
    pi_over_A = pi / A
    ks = ChainRulesCore.@ignore_derivatives pi_over_A * (-N .+ 2 * idx)
    # Interpolations cannot build the spline from device data: it is built on the host, and moved to the device
    # only when the strikes are there
    spline_cub = ChainRulesCore.@ignore_derivatives adapt_itp(CubicSplineInterpolation(ks, adapt(Array, integral_value)), StrikeVec)
    # map and not a broadcast: ReverseDiff tracks the coefficients of the spline, and cannot broadcast a function that
    # captures tracked values; on the device, map runs as a kernel
    prices = @. S0 * exp(dT) - exp(-rT) / pi * dx / 3 * sqrt(S0 * StrikeVec) * $map(spline_cub, log(StrikeVec / S0))
    return prices
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, opt, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    Npow = method.Npow
    N = ChainRulesCore.@ignore_derivatives 2^Npow
    S0 = mcProcess.underlying.S0
    A = method.A
    T = opt.T
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    correction = rT + dT - FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    idx = ChainRulesCore.@ignore_derivatives 0:(N-1)
    # one_minus_one = ChainRulesCore.@ignore_derivatives AlternateVector(Int8(1), Int8(-1), N)
    one_minus_one = ChainRulesCore.@ignore_derivatives PatternVector(N, EvenOddPattern(Int8(1), Int8(-1)))
    # weights_simpson = ChainRulesCore.@ignore_derivatives AlternatePaddedVector(Int8(1), Int8(4), Int8(2), Int8(1), N)
    weights_simpson = ChainRulesCore.@ignore_derivatives PatternVector(N, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
    support_array = ChainRulesCore.@ignore_derivatives @. one_minus_one * weights_simpson
    dx = A / N
    x = ChainRulesCore.@ignore_derivatives adapt_array(collect(idx * dx), mode)
    x_im = @. 1 // 2 + x * im
    Y = fft(@. support_array * exp(correction * x_im + FinancialFFT.characteristic_exponent_i(x_im, mcProcess, T)) / abs2(x_im))
    integral_value = @. real_mod(Y)
    pi_over_A = pi / A
    ks = ChainRulesCore.@ignore_derivatives pi_over_A * (-N .+ 2 * idx)
    StrikeVec = opt.K
    # Interpolations cannot build the spline from device data: it is built on the host, and moved to the device
    # only when the strikes are there
    spline_cub = ChainRulesCore.@ignore_derivatives adapt_itp(CubicSplineInterpolation(ks, adapt(Array, integral_value)), StrikeVec)
    # map and not a broadcast: ReverseDiff tracks the coefficients of the spline, and cannot broadcast a function that
    # captures tracked values; on the device, map runs as a kernel
    prices = @. S0 * exp(dT) - exp(-rT) / pi * dx / 3 * sqrt(S0 * StrikeVec) * $map(spline_cub, log(StrikeVec / S0))
    return call_to_put(prices, S0 * exp(dT), exp(-rT), opt)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, abstractPayoffs::Array{U}, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = abstractPayoffs[index_same_t]
        strikes = [opt.K for opt in payoffs]
        prices_call = pricer(mcProcess, strikes, zero_rate, T, method, mode)
        dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
        prices_final = [call_to_put(prices_call[i], mcProcess.underlying.S0 * exp(dT), exp(-FinancialMonteCarlo.integral(zero_rate.r, T)), payoffs[i]) for i in eachindex(payoffs)]
        prices[index_same_t] = prices_final
    end

    return prices
end