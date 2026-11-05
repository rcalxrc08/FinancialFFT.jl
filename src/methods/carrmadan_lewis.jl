"""
    CarrMadanLewisMethod(A, Npow)
    CarrMadanLewisMethod(ε; Npow_max = 23)

FFT evaluation of the Lewis (2001) formula: the integrand along the line `Im(u) = 1/2` is sampled
on `[0, A)` with `2^Npow` nodes and Simpson weights, a single FFT returns the integral on a log-strike grid,
and prices at the requested strikes are interpolated on it with six-point Lagrange polynomials.
Supports `EuropeanOption`, `BinaryEuropeanOption` and their smiles.

With a tolerance `ε`, `A` and `Npow` are chosen at pricing time so that the error of the prices is below `ε`, as for
[`CarrMadanMethod`](@ref).

# Arguments
- `A`: upper bound of the frequency integration domain (`A > 0`).
- `Npow`: base-2 logarithm of the number of FFT nodes (`2 < Npow < 24`).
- `ε`: tolerance on the prices (`ε > 0`); `Npow_max`: the largest `Npow` allowed.
"""
struct CarrMadanLewisMethod{G <: Union{FFTGrid, FFTTolerance}} <: AbstractFFTMethod
    grid::G
end
CarrMadanLewisMethod(A::Number, Npow::Integer) = CarrMadanLewisMethod(FFTGrid(A, Npow))
CarrMadanLewisMethod(ε::Real; Npow_max::Integer = 23) = CarrMadanLewisMethod(FFTTolerance(ε, Npow_max))

export CarrMadanLewisMethod;
# The binary call is minus the strike derivative of the Lewis formula: abs2(x_im) becomes x_im, and sqrt(S0 * K) becomes sqrt(S0 / K).
lewis_denominator(::VanillaPayoff) = abs2
lewis_denominator(::BinaryPayoff) = identity
carr_madan_lewis_call(integral, S0, dT, opt::VanillaPayoff) = @. S0 * exp(dT) - sqrt(S0 * opt.K) * integral
carr_madan_lewis_call(integral, S0, _, opt::BinaryPayoff) = @. sqrt(S0 / opt.K) * integral

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
    A_grid, Npow = ChainRulesCore.@ignore_derivatives fft_grid(method.grid, method, mcProcess, zero_rate, opt)
    N = ChainRulesCore.@ignore_derivatives 2^Npow
    S0 = mcProcess.underlying.S0
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    A = to_float_type(float_type(S0, rT, dT, T), A_grid)
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
    Y = fft_mod(@. support_array * exp(correction * x_im + FinancialFFT.characteristic_exponent_i(x_im, mcProcess, T)) / den(x_im))
    integral_value = @. real_mod(Y)
    # the integral on the log-strike grid -N π / A + 2 π j / A, j = 0, …, N - 1
    pi_over_A = pi / A
    integral = @. exp(-rT) * invπ * dx / 3 * $grid_interpolate(integral_value, -N * pi_over_A, twoπ / A, log(opt.K / S0))
    return call_to_put(carr_madan_lewis_call(integral, S0, dT, opt), S0 * exp(dT), exp(-rT), opt)
end
