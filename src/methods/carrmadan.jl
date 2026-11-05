"""
    CarrMadanMethod(A, Npow, α = 0.75)

Carr–Madan (1999) FFT method. The damped call transform is integrated with Simpson weights
on the frequency grid `v ∈ [0, A)` with `2^Npow` nodes, a single FFT returns call prices on a
log-strike grid, and prices at the requested strikes are interpolated on it with six-point Lagrange polynomials.
Supports `EuropeanOption`, `BinaryEuropeanOption` and their smiles.

# Arguments
- `A`: upper bound of the frequency integration domain (`A > 0`). The strike grid has step `2π / A`.
- `Npow`: base-2 logarithm of the number of FFT nodes (`2 < Npow < 24`).
- `α`: damping exponent (`α > 0`). It needs `E[S^(1 + α)]` to be finite, so it is capped halfway between 1 and the
  exponential moment bound of the model. The frequency step `A / 2^Npow` must be small with respect to the capped value.
"""
struct CarrMadanMethod{num <: Number, num_1 <: Integer, num_2 <: Number} <: AbstractFFTMethod
    A::num
    Npow::num_1
    α::num_2
    function CarrMadanMethod(A::num, N::num_1, α::num_2 = 0.75) where {num <: Number, num_1 <: Integer, num_2 <: Number}
        ChainRulesCore.@ignore_derivatives @assert(A > 0.0, "A must be positive")
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        ChainRulesCore.@ignore_derivatives @assert(N < 24, "N will cause overflow")
        ChainRulesCore.@ignore_derivatives @assert(α > 0.0, "α must be positive")
        return new{num, num_1, num_2}(A, N, α)
    end
end
export CarrMadanMethod;

# E[S^(1 + α)] must be finite: 1 + α stays at most halfway between 1 and the exponential moment bound of the upper tail.
carr_madan_damping(mcProcess, T, α) = min(α, convert(typeof(α), (exponential_moment_bound(mcProcess, 1, v_value_mod(T)) - 1) / 2))
# The binary call is minus the strike derivative of the call: α + iv leaves the denominator of the transform, and the
# price is divided by K.
carr_madan_factor(::VanillaPayoff) = identity
carr_madan_factor(::BinaryPayoff) = one
carr_madan_call(price, ::VanillaPayoff) = price
carr_madan_call(price, opt::BinaryPayoff) = @. price / opt.K

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanMethod, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    FT = float_type(S0, rT, dT, T)
    A = to_float_type(FT, method.A)
    #* v-> compute integral as a summation
    dx = A / N
    real_vec = 0:(N-1)
    v = ChainRulesCore.@ignore_derivatives adapt_array(collect(real_vec * dx), mode)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    one_adj = ChainRulesCore.@ignore_derivatives Int8(1)
    minus_one_adj = ChainRulesCore.@ignore_derivatives Int8(-1)
    one_minus_one = ChainRulesCore.@ignore_derivatives PatternVector(N, EvenOddPattern(one_adj, minus_one_adj))
    two_adj = ChainRulesCore.@ignore_derivatives Int8(2)
    four_adj = ChainRulesCore.@ignore_derivatives Int8(4)
    weights_ = ChainRulesCore.@ignore_derivatives PatternVector(N, PaddedEvenOddPattern(one_adj, four_adj, two_adj, one_adj))
    dk = twoπ / A
    # not pi * N / A: an irrational times an integer is a Float64
    b = N * (pi / A)
    drift_rd = rT - dT
    v_im = @. v * im
    alfa = ChainRulesCore.@ignore_derivatives carr_madan_damping(mcProcess, T, to_float_type(FT, method.α))
    den = carr_madan_factor(opt)
    # transform of exp(α k) C(k): E[exp((1 + α + iv) X)] / ((α + iv) (1 + α + iv)), i.e. the exponent evaluated in 1 + α + iv
    complex_vec_z_k = @. one_minus_one * cis_mod(drift_rd * v) * exp(FinancialFFT.characteristic_exponent_i(v_im + (1 + alfa), mcProcess, T) - (v_im + 1 + alfa) * corr) / (den(v_im + alfa) * (alfa + 1 + v_im)) * dx * weights_ / 3
    y = fft_mod(complex_vec_z_k)
    z_T = @. real_mod(y) * invπ

    # prices on the log-strike grid -b + dk j, j = 0, …, N - 1
    k_interp = @. log(opt.K / S0)
    prices = @. S0 * exp(-dT - alfa * (k_interp - drift_rd)) * $grid_interpolate(z_T, -b, dk, k_interp)
    return call_to_put(carr_madan_call(prices, opt), S0 * exp(-dT), exp(-rT), opt)
end
