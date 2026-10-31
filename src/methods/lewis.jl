"""
    LewisMethod(A, N)

Direct quadrature of the Lewis (2001) formula along the line `Im(u) = 1/2`: the integrand is evaluated
on `N` equally spaced nodes in `[-A, A]` and summed. No FFT and no interpolation are involved, so every
strike is priced exactly on its own and dual numbers, hyper-dual numbers, TaylorSeries, ForwardDiff, ReverseDiff, Zygote and Mooncake are supported.

# Arguments
- `A`: half-width of the frequency integration domain (`A > 0`).
- `N`: number of quadrature nodes (`N > 2`).
"""
struct LewisMethod{num <: Number, num_1 <: Integer} <: AbstractIntegralMethod
    A::num
    N::num_1
    function LewisMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(A > 0.0, "A must be positive")
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        return new{num, num_1}(A, N)
    end
end
export LewisMethod;
#Equivalent to real(exp(z)) but more efficient
exp_mod(x) = exp(real_mod(x)) * cos(imag_mod(x))

function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::FinancialMonteCarlo.EuropeanOption)
    corr_h = corr_adj / 2
    corr_im = corr_adj * im
    return @. FinancialFFT.exp_mod(corr_h + v * corr_im + char_exp_v) / (1 // 4 + v^2)
end

function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::EuropeanOptionSmile)
    return @. FinancialFFT.exp_mod((1 // 2 + v * im) * corr_adj' + char_exp_v) / (1 // 4 + v^2)
end

function evaluate_integrand_lewis_v2_scalar_binary(v, corr_adj, re1, im1)
    re_term = corr_adj / 2 + re1
    term_im = corr_adj * v + im1
    exp_re = exp(re_term)
    sin_im, cos_im = sincos(term_im)
    return exp_re * (cos_im / 2 + sin_im * v) / (1 // 4 + v^2)
end

# The real and imaginary parts are passed as separate arrays: Zygote cannot differentiate a GPU broadcast
# whose intermediate results are tuples.
function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::FinancialMonteCarlo.BinaryEuropeanOption)
    re_char_exp, im_char_exp = real_mod.(char_exp_v), imag_mod.(char_exp_v)
    return @. evaluate_integrand_lewis_v2_scalar_binary(v, corr_adj, re_char_exp, im_char_exp)
end

function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::FinancialMonteCarlo.BinaryEuropeanOptionSmile)
    re_char_exp, im_char_exp = real_mod.(char_exp_v), imag_mod.(char_exp_v)
    return @. evaluate_integrand_lewis_v2_scalar_binary(v, corr_adj', re_char_exp, im_char_exp)
end

# function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::FinancialMonteCarlo.BinaryEuropeanOption)
#     corr_h = corr_adj / 2
#     corr_im = corr_adj * im
#     return @. FinancialFFT.real_mod(exp(corr_h + v * corr_im + char_exp_v) / (1 // 2 + im * v))
# end

function convert_integral_result_to_price(discounted_sum_, _, _, df, opt::BinaryEuropeanOption)
    return df * ifelse(opt.isCall, discounted_sum_, 1 - discounted_sum_)
end

function convert_integral_result_to_price(discounted_sum, S0, dT, df, opt::EuropeanOption)
    S0_adj = S0 * exp(dT) / df
    diff_typed = S0_adj - opt.K
    return df * (ifelse(opt.isCall, diff_typed, zero(diff_typed)) + opt.K * (1 - discounted_sum))
end

function convert_integral_result_to_price_interf(dx_adj, discounted_sum, S0, dT, df, opt::EuropeanOptionSmile)
    S0_adj = S0 * exp(dT) / df
    zero_typed = zero(S0_adj + zero(eltype(opt.K)))
    return @. df * (ifelse(opt.isCall, S0_adj - opt.K, zero_typed) + opt.K * (1 - dx_adj * discounted_sum / pi))
end

function convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, opt::BinaryEuropeanOptionSmile)
    #simplify the following
    discounted_sum = @. dx_adj * total_sum_lewis / π
    return @. df * ifelse(opt.isCall, discounted_sum, 1 - discounted_sum)
end

function convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, opt)
    #simplify the following
    discounted_sum = dx_adj * total_sum_lewis / π
    convert_integral_result_to_price(discounted_sum, S0, dT, df, opt)
end

struct LewisMethodResult{num_1, num_2}
    char_exp_v::num_1
    v_im_adj::num_2
    function LewisMethodResult(a::num_1, b::num_2) where {num_1, num_2}
        return new{num_1, num_2}(a, b)
    end
end

function lw_integrate(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
    corr_adj = log(S0 / abstractPayoff.K) + rT_corr + dT
    char_exp_v = char_exp_v_r.char_exp_v
    v_im_adj = char_exp_v_r.v_im_adj
    total_sum_lewis = sum(evaluate_integrand_lewis_v2(v_im_adj, corr_adj, char_exp_v, abstractPayoff))
    return convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, abstractPayoff)
end

function lw_integrate(char_exp_v_r, abstractPayoff::EuropeanOptionSmile, df, dT, dx_adj, S0, rT_corr)
    corr_adj = @. log(S0 / abstractPayoff.K) + rT_corr + dT
    char_exp_v = char_exp_v_r.char_exp_v
    v_im_adj = char_exp_v_r.v_im_adj
    total_sum_lewis = sum(evaluate_integrand_lewis_v2(v_im_adj, corr_adj, char_exp_v, abstractPayoff), dims = 1)'
    return convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, abstractPayoff)
end

function lw_integrate(char_exp_v_r, abstractPayoff::BinaryEuropeanOptionSmile, df, dT, dx_adj, S0, rT_corr)
    corr_adj = @. log(S0 / abstractPayoff.K) + rT_corr + dT
    char_exp_v = char_exp_v_r.char_exp_v
    v_im_adj = char_exp_v_r.v_im_adj
    total_sum_lewis = sum(evaluate_integrand_lewis_v2(v_im_adj, corr_adj, char_exp_v, abstractPayoff), dims = 1)'
    return convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, abstractPayoff)
end

"""
    pricer(mcProcess, zero_rate, method, payoff[, mode])

Prices `payoff` under `mcProcess` with the Fourier method `method`
(`LewisMethod`, `CosMethod`, `CarrMadanLewisMethod`, `CarrMadanMethod` or `DensityInverter`).

# Arguments
- `mcProcess`: a `FinancialMonteCarlo.BaseProcess`, e.g. `BlackScholesProcess(σ, Underlying(S0, d))`.
- `zero_rate`: a `FinancialMonteCarlo.AbstractZeroRateCurve`, e.g. `ZeroRate(r)`.
- `method`: the configuration of the numerical method.
- `payoff`: an `EuropeanOption` or a `BinaryEuropeanOption`, a smile (`EuropeanOptionSmile`,
  `BinaryEuropeanOptionSmile`, several strikes sharing the same maturity). Options with different maturities are
  priced by broadcasting, `pricer.(mcProcess, zero_rate, method, payoffs)`.
- `mode`: `FinancialMonteCarlo.SerialMode()` (default) or a CUDA mode when CUDA.jl is loaded.

Returns a price, or a vector of prices for smiles.
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FinancialFFT.LewisMethod, abstractPayoff::FourierPayoff, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = abstractPayoff.T
    A = method.A
    N = method.N
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    eps_typed = ChainRulesCore.@ignore_derivatives eps(zero(Float64))
    range_init = ChainRulesCore.@ignore_derivatives adapt_array(collect(range(-1, length = N, stop = 1)), mode)
    one_half = 1 // 2
    v = ChainRulesCore.@ignore_derivatives @. (eps_typed + A * range_init)
    # v_im_adj = @. one_half + v * im
    char_exp_v = @. FinancialFFT.characteristic_exponent_i(one_half + v * im, mcProcess, T)
    lw_result = LewisMethodResult(char_exp_v, v)
    dx_adj = A / (N - 1)
    df = exp(-rT)
    rT_corr = rT - corr
    return lw_integrate(lw_result, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
end
