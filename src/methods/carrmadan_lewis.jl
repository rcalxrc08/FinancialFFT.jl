"""
    CarrMadanLewisMethod(A, Npow)

FFT evaluation of the Lewis (2001) formula: the integrand along the line `Im(u) = 1/2` is sampled
on `[0, A)` with `2^Npow` nodes and Simpson weights, a single FFT returns the integral on a log-strike grid,
and prices at the requested strikes are obtained through cubic-spline interpolation.
Supports `EuropeanOption`, `BinaryEuropeanOption` and their smiles.

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
# The binary call is minus the strike derivative of the Lewis formula: abs2(x_im) becomes x_im, and sqrt(S0 * K) becomes sqrt(S0 / K).
lewis_denominator(::VanillaPayoff) = abs2
lewis_denominator(::BinaryPayoff) = identity
carr_madan_lewis_call(integral, S0, dT, opt::VanillaPayoff) = @. S0 * exp(dT) - sqrt(S0 * opt.K) * integral
carr_madan_lewis_call(integral, S0, _, opt::BinaryPayoff) = @. sqrt(S0 / opt.K) * integral

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
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
    den = lewis_denominator(opt)
    Y = fft(@. support_array * exp(correction * x_im + FinancialFFT.characteristic_exponent_i(x_im, mcProcess, T)) / den(x_im))
    integral_value = @. real_mod(Y)
    pi_over_A = pi / A
    ks = ChainRulesCore.@ignore_derivatives pi_over_A * (-N .+ 2 * idx)
    # Interpolations cannot build the spline from device data: it is built on the host, and moved to the device
    # only when the strikes are there
    spline_cub = ChainRulesCore.@ignore_derivatives adapt_itp(CubicSplineInterpolation(ks, adapt(Array, integral_value)), opt.K)
    # map and not a broadcast: ReverseDiff tracks the coefficients of the spline, and cannot broadcast a function that
    # captures tracked values; on the device, map runs as a kernel
    integral = @. exp(-rT) / pi * dx / 3 * $map(spline_cub, log(opt.K / S0))
    return call_to_put(carr_madan_lewis_call(integral, S0, dT, opt), S0 * exp(dT), exp(-rT), opt)
end
