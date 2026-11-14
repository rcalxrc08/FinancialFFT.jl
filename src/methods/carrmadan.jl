# The grid of the Carr-Madan methods: given (A, Npow), or computed at pricing time from a price tolerance, when the model
# and the strikes are known (fft_parameters.jl). fft_grid returns (A, Npow).
struct FFTGrid{num <: Number, num_1 <: Integer}
    A::num
    Npow::num_1
    function FFTGrid(A::num, Npow::num_1) where {num <: Number, num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(A > 0.0, "A must be positive")
        ChainRulesCore.@ignore_derivatives @assert(Npow > 2, "Npow must be greater than 2")
        ChainRulesCore.@ignore_derivatives @assert(Npow < 24, "Npow will cause overflow")
        return new{num, num_1}(A, Npow)
    end
end
struct FFTTolerance{num <: Real, num_1 <: Integer}
    ε::num
    Npow_max::num_1
    function FFTTolerance(ε::num, Npow_max::num_1) where {num <: Real, num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(ε > 0, "the tolerance must be positive")
        ChainRulesCore.@ignore_derivatives @assert(2 < Npow_max < 24, "Npow_max must be between 3 and 23")
        return new{num, num_1}(ε, Npow_max)
    end
end
fft_grid(grid::FFTGrid, method, mcProcess, zero_rate, opt) = (grid.A, grid.Npow)

"""
    CarrMadanMethod(A, Npow, α = 0.75; mode = SerialMode())
    CarrMadanMethod(ε; α = 0.75, Npow_max = 23, mode = SerialMode())

Carr–Madan (1999) FFT method. The damped call transform is integrated with Simpson weights
on the frequency grid `v ∈ [0, A)` with `2^Npow` nodes, a single FFT returns call prices on a
log-strike grid, and prices at the requested strikes are interpolated on it with six-point Lagrange polynomials.
Supports `EuropeanOption`, `BinaryEuropeanOption` and their smiles.

With a tolerance `ε`, `A` and `Npow` are chosen at pricing time, for the model and the strikes of the payoff, so that the
error of the prices is below `ε`: from bounds on the interpolation, truncation and aliasing errors, computed in the float
type of the inputs. The bounds do not cover rounding (`ε` must be above the rounding floor of that type), the
parameters are constants for differentiation, and pricing throws an `ArgumentError` when more than `2^Npow_max` nodes are
needed. Choosing them costs about one price: for a model priced many times, compute them once and pass `(A, Npow)`.

# Arguments
- `A`: upper bound of the frequency integration domain (`A > 0`). The strike grid has step `2π / A`.
- `Npow`: base-2 logarithm of the number of FFT nodes (`2 < Npow < 24`).
- `α`: damping exponent (`α > 0`). It needs `E[S^(1 + α)]` to be finite, so it is capped halfway between 1 and the
  exponential moment bound of the model. The frequency step `A / 2^Npow` must be small with respect to the capped value.
- `ε`: tolerance on the prices (`ε > 0`); `Npow_max`: the largest `Npow` allowed.
- `mode` (keyword): the execution mode, `FinancialMonteCarlo.SerialMode()` (default) or `FinancialMonteCarlo.CudaMode()` with CUDA.jl loaded; `FinancialMonteCarlo.with_mode(method, mode)` changes it.
"""
struct CarrMadanMethod{G <: Union{FFTGrid, FFTTolerance}, num_2 <: Number, modeType <: FinancialMonteCarlo.AbstractMode} <: AbstractFFTMethod
    grid::G
    α::num_2
    mode::modeType
    function CarrMadanMethod(grid::G, α::num_2, mode::modeType) where {G <: Union{FFTGrid, FFTTolerance}, num_2 <: Number, modeType <: FinancialMonteCarlo.AbstractMode}
        ChainRulesCore.@ignore_derivatives @assert(α > 0.0, "α must be positive")
        return new{G, num_2, modeType}(grid, α, mode)
    end
end
CarrMadanMethod(A::Number, Npow::Integer, α::Number = 0.75; mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = CarrMadanMethod(FFTGrid(A, Npow), α, mode)
CarrMadanMethod(ε::Real; α::Number = 0.75, Npow_max::Integer = 23, mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = CarrMadanMethod(FFTTolerance(ε, Npow_max), α, mode)
export CarrMadanMethod;

# E[S^(1 + α)] must be finite: 1 + α stays at most halfway between 1 and the exponential moment bound of the upper tail.
carr_madan_damping(mcProcess, T, α) = min(α, convert(typeof(α), (exponential_moment_bound(mcProcess, 1, v_value_mod(T)) - 1) / 2))
# The binary call is minus the strike derivative of the call: α + iv leaves the denominator of the transform, and the
# price is divided by K.
carr_madan_factor(::VanillaPayoff) = identity
carr_madan_factor(::BinaryPayoff) = one
carr_madan_call(price, ::VanillaPayoff) = price
carr_madan_call(price, opt::BinaryPayoff) = @. price / opt.K
# 1 / z as conj(z) / |z|², with one real division: z is a denominator of the transforms, far from 0 and from overflow.
inv_denominator(z) = conj(z) * inv(abs2(z))
# Quadrature: Simpson's rule on the nodes v_j = j dx, weights (1, 4, 2, 4, …, 4, 2, 1) dx / 3 (2^Npow nodes, an even number:
# the last panel is not closed as Simpson's, harmless where the integrand has decayed), error O(dx⁴) on the
# transform. It is not what limits the prices: the log-strike step 2π / A, fixed by A whatever Npow, sets the error of the
# 6-point interpolation in the strike (#88).
# The constant weights of the Carr-Madan transform on the nodes v: signs (-1)^j, Simpson weights, dx / 3 and
# 1 / ((α + iv) (1 + α + iv)), with α + iv dropped by den = one for a binary. In a loop of their own, which vectorises:
# faster, as measured, than computing them in the broadcast of the exponential.
function carr_madan_weights(v, alfa, dx, den)
    N = length(v)
    one_minus_one = PatternVector(N, EvenOddPattern(Int8(1), Int8(-1)))
    simpson = PatternVector(N, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
    dx_3 = dx / 3
    return @. one_minus_one * simpson * dx_3 * inv_denominator(den(v * im + alfa) * (alfa + 1 + v * im))
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanMethod, opt_::FourierPayoff)
    mode = FinancialMonteCarlo.execution_mode(method)
    # the strikes of a smile go to the device of the mode, wherever they are
    opt = FinancialMonteCarlo.to_mode(opt_, mode)
    T = opt.T
    A_grid, Npow = ChainRulesCore.@ignore_derivatives fft_grid(method.grid, method, mcProcess, zero_rate, opt)
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    FT = float_type(S0, rT, dT, T)
    A = to_float_type(FT, A_grid)
    #* v-> compute integral as a summation
    dx = A / N
    # A vector, not a float range: Enzyme, which ignores @ignore_derivatives, fails on the TwicePrecision of a float range even
    # with constant inputs (EnzymeAD/Enzyme.jl#4016; Mooncake with active ones, see fft_density.jl). k * dx is rounded once, as the range is.
    v = ChainRulesCore.@ignore_derivatives adapt_array([k * dx for k = 0:(N-1)], mode)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    dk = twoπ / A
    # not pi * N / A: an irrational times an integer is a Float64
    b = N * (pi / A)
    drift_rd = rT - dT
    alfa = ChainRulesCore.@ignore_derivatives carr_madan_damping(mcProcess, T, to_float_type(FT, method.α))
    # transform of exp(α k) C(k): E[exp((1 + α + iv) X)] / ((α + iv) (1 + α + iv)), i.e. the exponent evaluated in
    # 1 + α + iv. The factors that do not depend on the model are constant weights (the signs (-1)^j that centre the strike
    # grid, the Simpson weights, dx / 3, the inverse of the denominator with one real division), on floats even when the
    # exponent is a dual.
    weights = ChainRulesCore.@ignore_derivatives carr_madan_weights(v, alfa, dx, carr_madan_factor(opt))
    # exponent ψ(1 + α + iv) - (1 + α + iv) corr + i v drift: its real constant -(1 + α) corr leaves the FFT (which is
    # linear) and joins the factor of the prices
    drift_corr = drift_rd - corr
    complex_vec_z_k = @. weights * exp(FinancialFFT.characteristic_exponent_i(v * im + (1 + alfa), mcProcess, T) + v * im * drift_corr)
    y = fft(complex_vec_z_k)
    z_T = @. real_mod(y) * invπ

    # prices on the log-strike grid -b + dk j, j = 0, …, N - 1
    k_interp = @. log(opt.K / S0)
    prices = @. S0 * exp(-dT - (1 + alfa) * corr - alfa * (k_interp - drift_rd)) * $grid_interpolate(z_T, -b, dk, k_interp)
    return call_to_put(carr_madan_call(prices, opt), S0 * exp(-dT), exp(-rT), opt)
end
