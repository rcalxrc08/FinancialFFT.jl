"""
    DensityInverter(N = 14, M = 2^10)

Prices by integrating the payoff against the density of [`FFTDensityInverter`](@ref)`(N)`: the characteristic
function is inverted by a single FFT on `2^N` points of the range of `CosMethod`, and the put is integrated by
Simpson's rule on `M` intervals of log-moneyness ending at the strike, so that the kink of the payoff is an endpoint;
the call follows by parity, as in [`CosDensityMethod`](@ref). The range comes from the model, so no model needs a
range parameter. Supports `EuropeanOption`, `BinaryEuropeanOption` and their smiles: the strikes of a smile share the
density, and their prices are the ones of the single options.

# Arguments
- `N`: base-2 logarithm of the number of points of the density (`N > 2`).
- `M`: number of Simpson intervals for each strike (even, `M > 2`).

`DensityInverter(N, xmax)` is deprecated: the range of the density no longer depends on `xmax`.
"""
struct DensityInverter{num_1 <: Integer, num_2 <: Integer} <: AbstractIntegralMethod
    N::num_1
    M::num_2
    function DensityInverter(N::num_1 = 14, M::num_2 = 2^10) where {num_1 <: Integer, num_2 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        ChainRulesCore.@ignore_derivatives @assert(M > 2 && iseven(M), "M must be even and greater than 2")
        return new{num_1, num_2}(N, M)
    end
end

function DensityInverter(N::Integer, ::AbstractFloat)
    Base.depwarn("`DensityInverter(N, xmax)` is deprecated: the density spans the range of `CosMethod`, computed from the model, and `xmax` is ignored. Use `DensityInverter(N)`.", :DensityInverter)
    return DensityInverter(N)
end

# The floor gets the type of the value from the value itself: the type of a TaylorN says neither its order nor its space.
density_floor(y, eps_mod) = max(y, zero(y) + eps_mod)

# Density of log(S_T / S0) + shift on the 2^N points -A + 2A j / 2^N, j = 0, …, 2^N - 1, one column per shift: the
# characteristic function is sampled on the frequencies π k / A, k = -2^(N-1), …, 2^(N-1) - 1, and the alternating
# signs (-1)^k and (-1)^j center both grids. The sign (-1)^j is dropped with abs, and the values are floored at eps.
function fft_density_values(mcProcess, r, N, A, shift, T, mode)
    n = ChainRulesCore.@ignore_derivatives 2^N
    vec_lin = ChainRulesCore.@ignore_derivatives 0:(n-1)
    vec = ChainRulesCore.@ignore_derivatives adapt_array(collect(-div(n, 2) .+ vec_lin), mode)            # Indices
    drift_rd = FinancialMonteCarlo.integral(r.r - mcProcess.underlying.d, T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    one_minus_one = ChainRulesCore.@ignore_derivatives PatternVector(n, EvenOddPattern(Int8(1), Int8(-1)))
    drift_rf = @. drift_rd + shift - corr
    dt = pi / A # Step size, frequency space
    dt_im = dt * im
    drift_dt_im = strike_columns(drift_rf .* dt_im)
    char_exp = characteristic_exponent_i.(vec .* dt_im, mcProcess, T)
    Y_r = fft_columns(@. exp(char_exp + drift_dt_im * vec) * one_minus_one)
    twice_A = 2 * A
    eps_mod = ChainRulesCore.@ignore_derivatives eps(v_value_mod(zero(eltype(drift_rf))))
    density_vals_adj = density_floor.(abs_mod.(FinancialFFT.real_mod.(Y_r)) ./ twice_A, eps_mod)
    ChainRulesCore.@ignore_derivatives @assert !isnan(sum(density_vals_adj) * twice_A / n) "Returned result is NaN"
    return density_vals_adj
end

function adjust_price_adimensional(p, euPayoff::EuropeanOption)
    K = euPayoff.K
    return K * p
end
adjust_price_adimensional(V, ::BinaryEuropeanOption) = V
adjust_price_adimensional(p, opt::EuropeanOptionSmile) = opt.K .* p
adjust_price_adimensional(V, ::BinaryEuropeanOptionSmile) = V

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::DensityInverter, opt::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    f = density(mcProcess, r, FFTDensityInverter(method.N), opt.T, mode)
    return density_integral_price(mcProcess, r, f, method.M, opt, mode)
end

function density(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::DensityInverter, T, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    Base.depwarn("`density` with `DensityInverter` is deprecated: use `FFTDensityInverter(N)`, which it now returns.", :density)
    return density(mcProcess, zero_rate, FFTDensityInverter(method.N), T, mode)
end
