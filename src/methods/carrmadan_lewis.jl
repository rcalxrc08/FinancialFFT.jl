"""
    CarrMadanLewisMethod(A, Npow; mode = SerialMode())
    CarrMadanLewisMethod(ε; Npow_max = 23, mode = SerialMode())

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
- `mode` (keyword): the execution mode, `FinancialMonteCarlo.SerialMode()` (default) or `FinancialMonteCarlo.CudaMode()` with CUDA.jl loaded; `FinancialMonteCarlo.with_mode(method, mode)` changes it.
"""
struct CarrMadanLewisMethod{G <: Union{FFTGrid, FFTTolerance}, modeType <: FinancialMonteCarlo.AbstractMode} <: AbstractFFTMethod
    grid::G
    mode::modeType
end
CarrMadanLewisMethod(A::Number, Npow::Integer; mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = CarrMadanLewisMethod(FFTGrid(A, Npow), mode)
CarrMadanLewisMethod(ε::Real; Npow_max::Integer = 23, mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = CarrMadanLewisMethod(FFTTolerance(ε, Npow_max), mode)

export CarrMadanLewisMethod;
# The binary call is minus the strike derivative of the Lewis formula: abs2(x_im) becomes x_im, and sqrt(S0 * K) becomes sqrt(S0 / K).
lewis_denominator(::VanillaPayoff) = abs2
lewis_denominator(::BinaryPayoff) = identity
carr_madan_lewis_call(integral, S0, dT, opt::VanillaPayoff) = @. S0 * exp(dT) - sqrt(S0 * opt.K) * integral
carr_madan_lewis_call(integral, S0, _, opt::BinaryPayoff) = @. sqrt(S0 / opt.K) * integral
# 1 / den(1/2 + ix), a function of x chosen by the payoff, so that the payoff (which holds host arrays) stays out of device
# kernels: a real division for a vanilla option, one real division (inv_denominator) for a binary.
vanilla_lewis_inverse_denominator(x) = inv(1 // 4 + x^2)
binary_lewis_inverse_denominator(x) = inv_denominator(1 // 2 + x * im)
lewis_inverse_denominator(::VanillaPayoff) = vanilla_lewis_inverse_denominator
lewis_inverse_denominator(::BinaryPayoff) = binary_lewis_inverse_denominator

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, opt_::FourierPayoff)
    mode = FinancialMonteCarlo.execution_mode(method)
    opt = FinancialMonteCarlo.to_mode(opt_, mode)
    T = opt.T
    A_grid, Npow = ChainRulesCore.@ignore_derivatives fft_grid(method.grid, method, mcProcess, zero_rate, opt)
    N = ChainRulesCore.@ignore_derivatives 2^Npow
    S0 = mcProcess.underlying.S0
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    A = to_float_type(float_type(S0, rT, dT, T), A_grid)
    correction = rT + dT - FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    dx = A / N
    # A vector, not a float range: Enzyme, which ignores @ignore_derivatives, fails on the TwicePrecision of a float range even
    # with constant inputs (Mooncake with active ones, see fft_density.jl). k * dx is rounded once, as the range is.
    x = ChainRulesCore.@ignore_derivatives adapt_array([k * dx for k = 0:(N-1)], mode)
    # Quadrature: Simpson's rule on the nodes x_j = j dx, weights (1, 4, 2, 4, …, 4, 2, 1) dx / 3 (an even number of
    # nodes: the last panel is not closed as Simpson's, harmless where the integrand has decayed), error O(dx⁴); as in
    # Carr-Madan, the strike interpolation on the step 2π / A limits the prices, not this rule (#88).
    # The factors that do not depend on the model (signs (-1)^j, Simpson weights, dx / 3, 1 / den(1/2 + ix) on floats) are
    # computed in the same broadcast. The nodes are complex(one_half, x), with a float one_half: a Rational added to a
    # complex number in this lazy argument (1 // 2 + x * im) makes the vectorised exponent 2.3 times slower.
    one_minus_one, simpson = ChainRulesCore.@ignore_derivatives PatternVector(N, EvenOddPattern(Int8(1), Int8(-1))), PatternVector(N, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
    inverse_denominator = lewis_inverse_denominator(opt)
    dx_3 = dx / 3
    one_half = one(dx) / 2
    # correction (1/2 + ix) + ψ(1/2 + ix): the real constant correction / 2 leaves the FFT and joins the factor of the integral
    Y = fft(@. one_minus_one * simpson * dx_3 * inverse_denominator(x) * exp(FinancialFFT.characteristic_exponent_i(complex(one_half, x), mcProcess, T) + x * im * correction))
    integral_value = @. real_mod(Y)
    # the integral on the log-strike grid -N π / A + 2 π j / A, j = 0, …, N - 1
    pi_over_A = pi / A
    integral = @. exp(correction / 2 - rT) * invπ * $grid_interpolate(integral_value, -N * pi_over_A, twoπ / A, log(opt.K / S0))
    return call_to_put(carr_madan_lewis_call(integral, S0, dT, opt), S0 * exp(dT), exp(-rT), opt)
end
