# The nodes of LewisMethod: given, or chosen at pricing time from a price tolerance (#48).
struct QuadratureGrid{num <: Number, num_1 <: Integer}
    A::num
    N::num_1
    function QuadratureGrid(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        check_lewis(A, N)
        return new{num, num_1}(A, N)
    end
end
struct QuadratureTolerance{num <: Real, num_1 <: Integer}
    ε::num
    N_max::num_1
    function QuadratureTolerance(ε::num, N_max::num_1) where {num <: Real, num_1 <: Integer}
        check_quadrature_tolerance(ε, N_max)
        return new{num, num_1}(ε, N_max)
    end
end
quadrature_grid(grid::QuadratureGrid, method, mcProcess, und, zero_rate, opt) = (grid.A, grid.N)

"""
    LewisMethod(A, N; mode = SerialMode())
    LewisMethod(ε; N_max = 2^20 + 1, mode = SerialMode())

Direct quadrature of the Lewis (2001) formula along the line `Im(u) = 1/2`: the integrand is evaluated
on `N` equally spaced nodes in `[-A, A]` and summed. No FFT and no interpolation are involved, so every
strike is priced exactly on its own and dual numbers, hyper-dual numbers, TaylorSeries, ForwardDiff, ReverseDiff, Zygote and Mooncake are supported.

With a tolerance `ε` (a float), `A` and `N` are chosen at pricing time, for the model, the maturity and the strikes,
so that the price error stays below `ε`; both come from the characteristic exponent:
- `A` (truncation): the tail beyond `A` of the modulus of the integrand, `|φ(v - i/2)|` times the Lewis weights,
  integrated numerically at the parameters of the model, gets `ε / 2`. It is an estimate (a quadrature of the modulus
  on `[1e-3, 1e6]`), not a rigorous bound.
- the step `2A / (N - 1)` (aliasing): the quadrature adds copies of the price at log-strike distances that are multiples of
  `L = 2π (N - 1) / (2A)`; for vanilla options they are bounded in closed form (no model), for binaries one side needs a
  Chernoff bound, i.e. the moment generating function `E[exp(p X_T)]` at real `p`. Each side gets `ε / 4`.
The choice evaluates the exponent 2048 times, about one and a half prices with the grid it picks (a Black–Scholes smile:
90 μs against 60 μs); for repeated pricing (a calibration),
resolve it once with `FinancialFFT.quadrature_grid` and pass `LewisMethod(A, N)`. The bounds do not cover the
rounding: `ε` must stay above the floor of the formula, about `eps() * max(S0, K)` (#40, #99).

# Arguments
- `A`: half-width of the frequency integration domain (`A > 0`).
- `N`: number of quadrature nodes (`N > 2`).
- `ε`: the price tolerance, an `AbstractFloat` (`ε > 0`): an integer argument is never read as a tolerance.
- `N_max` (keyword): the largest number of nodes the tolerance may ask for; more throws an `ArgumentError`.
- `mode` (keyword): the execution mode, `FinancialMonteCarlo.SerialMode()` (default) or `FinancialMonteCarlo.CudaMode()` with CUDA.jl loaded; `FinancialMonteCarlo.with_mode(method, mode)` changes it.
"""
struct LewisMethod{G <: Union{QuadratureGrid, QuadratureTolerance}, modeType <: FinancialMonteCarlo.AbstractMode} <: AbstractIntegralMethod
    grid::G
    mode::modeType
    function LewisMethod(grid::G, mode::modeType) where {G <: Union{QuadratureGrid, QuadratureTolerance}, modeType <: FinancialMonteCarlo.AbstractMode}
        return new{G, modeType}(grid, mode)
    end
end
LewisMethod(A::Number, N::Integer, mode::FinancialMonteCarlo.AbstractMode) = LewisMethod(QuadratureGrid(A, N), mode)
LewisMethod(A::Number, N::Integer; mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = LewisMethod(QuadratureGrid(A, N), mode)
LewisMethod(ε::AbstractFloat; N_max::Integer = 2^20 + 1, mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = LewisMethod(QuadratureTolerance(ε, N_max), mode)
export LewisMethod;

# quadrature_grid(method.grid, method, model, underlying, zero_rate, payoff): the half-width A and the number of nodes N
# that method uses for this model and payoff, the given ones or those its tolerance gives; LewisMethod(quadrature_grid(...)...)
# prices with them without choosing them again.
function quadrature_grid(grid::QuadratureTolerance, ::LewisMethod, model, und, zero_rate, opt)
    setup = lewis_error_setup(model, und, zero_rate, opt, grid.ε)
    A_min = truncation_frequency_bound(setup.modulus, setup.scale, setup.ε / 2, setup.ε)
    # The rule is trapezoidal (weight 1 on every node), so its copies sit at the multiples of L = 2π / h, h = 2A / (N - 1),
    # with weight 1, and none at half the period as for Simpson's rule. Weight 2 on the first copy covers the others: for
    # a vanilla option they decrease geometrically, by exp(-L / 2) < 1/2.
    L_min = lewis_aliasing_distance(setup, model, opt, 2 * one(setup.FT))
    # constants for differentiation (ReverseDiff and Mooncake do not honor @ignore_derivatives), before the integer conversion
    A_min, L_min = ignore_derivatives_mod(A_min), ignore_derivatives_mod(L_min)
    # h <= 2π / L_min and A >= A_min with an odd N (the node 0 keeps its weight); the spare nodes go to A
    half = ceil(Int, A_min * L_min / twoπ)
    N = 2 * half + 1
    N > grid.N_max && throw(ArgumentError("a tolerance of $(grid.ε) needs $N nodes, more than N_max = $(grid.N_max)"))
    # not pi * half: an irrational times an integer is a Float64
    return half * (twoπ / L_min), N
end

function evaluate_integrand_lewis_v2(v, corr_adj, char_exp_v, ::FinancialMonteCarlo.EuropeanOption)
    corr_h = corr_adj / 2
    corr_im = corr_adj * im
    return @. FinancialFFT.exp_mod(corr_h + v * corr_im + char_exp_v) / (1 // 4 + v^2)
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

function convert_integral_result_to_price(discounted_sum_, _, _, df, opt::BinaryEuropeanOption)
    return df * ifelse(opt.isCall, discounted_sum_, 1 - discounted_sum_)
end

function convert_integral_result_to_price(discounted_sum, S0, dT, df, opt::EuropeanOption)
    S0_adj = S0 * exp(dT) / df
    diff_typed = S0_adj - opt.K
    return df * @muladd(ifelse(opt.isCall, diff_typed, zero(diff_typed)) + opt.K * (1 - discounted_sum))
end

function convert_integral_result_to_price_interf(dx_adj, discounted_sum, S0, dT, df, opt::EuropeanOptionSmile)
    S0_adj = S0 * exp(dT) / df
    zero_typed = zero(S0_adj + zero(eltype(opt.K)))
    return @. df * (ifelse(opt.isCall, S0_adj - opt.K, zero_typed) + opt.K * (1 - dx_adj * discounted_sum * invπ))
end

function convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, opt::BinaryEuropeanOptionSmile)
    #simplify the following
    discounted_sum = @. dx_adj * total_sum_lewis * invπ
    return @. df * ifelse(opt.isCall, discounted_sum, 1 - discounted_sum)
end

function convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, opt)
    #simplify the following
    discounted_sum = dx_adj * total_sum_lewis * invπ
    convert_integral_result_to_price(discounted_sum, S0, dT, df, opt)
end

# The characteristic exponent on the nodes v >= 0 of the quadrature, and their weights: the integrand is even in v
# (ψ(1/2 - iv) is the conjugate of ψ(1/2 + iv)), so each node stands for itself and its opposite, the node 0 (odd N) for itself only.
struct LewisMethodResult{num_1, num_2, num_3}
    char_exp_v::num_1
    v_im_adj::num_2
    weights::num_3
    function LewisMethodResult(a::num_1, b::num_2, c::num_3) where {num_1, num_2, num_3}
        return new{num_1, num_2, num_3}(a, b, c)
    end
end

function lw_integrate(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr)
    corr_adj = log(S0 / abstractPayoff.K) + rT_corr + dT
    char_exp_v = char_exp_v_r.char_exp_v
    v_im_adj = char_exp_v_r.v_im_adj
    total_sum_lewis = sum(char_exp_v_r.weights .* evaluate_integrand_lewis_v2(v_im_adj, corr_adj, char_exp_v, abstractPayoff))
    return convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, abstractPayoff)
end

# A smile outside SerialMode (on the GPU), as one broadcast over nodes and strikes. With c_k the log-moneyness term of
# strike k, the integrand is exp(c_k / 2) times a_j cos(v_j c_k + b_j), so exp(c_k / 2) leaves the sum over the nodes and
# each term costs one cos (issue #46):
# - vanilla: exp_mod((1/2 + iv) c + ψ) / (1/4 + v²) = exp(c / 2) exp(Re ψ) cos(v c + Im ψ) / (1/4 + v²);
# - binary: exp(c / 2 + Re ψ) (cos θ / 2 + v sin θ) / (1/4 + v²), θ = v c + Im ψ, and
#   cos θ / 2 + v sin θ = sqrt(1/4 + v²) cos(θ - atan(2v)).
# Everything is real: no tuples in the broadcast (Zygote on the GPU), and no arrays mixing real and complex numbers.
lewis_amplitudes_phases(v, char_exp_v, weights, ::VanillaPayoff) = (@. weights * exp(real_mod(char_exp_v)) / (1 // 4 + v^2)), imag_mod.(char_exp_v)
lewis_amplitudes_phases(v, char_exp_v, weights, ::BinaryPayoff) = (@. weights * exp(real_mod(char_exp_v)) / sqrt(1 // 4 + v^2)), (@. imag_mod(char_exp_v) - atan(2 * v))
function lw_integrate(char_exp_v_r, abstractPayoff::SmilePayoff, df, dT, dx_adj, S0, rT_corr)
    corr_adj = @. log(S0 / abstractPayoff.K) + rT_corr + dT
    v = char_exp_v_r.v_im_adj
    a, b = lewis_amplitudes_phases(v, char_exp_v_r.char_exp_v, char_exp_v_r.weights, abstractPayoff)
    total_sum_lewis = exp.(corr_adj ./ 2) .* phased_cos_sum(a, b, v, corr_adj)
    return convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, abstractPayoff)
end

lw_integrate(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr, ::FinancialMonteCarlo.AbstractMode) = lw_integrate(char_exp_v_r, abstractPayoff, df, dT, dx_adj, S0, rT_corr)

# Coefficients C_j of the integrand, Re[exp((1/2 + i v_j) c) C_j] for c = log(S0 / K) + …, as real and imaginary parts:
# exp(ψ_j) / (1/4 + v_j²) for vanilla options, exp(ψ_j) / (1/2 + i v_j) for binary ones.
function lewis_coefficients(v, char_exp_v, ::VanillaPayoff)
    w = @. exp(real_mod(char_exp_v)) / (1 // 4 + v^2)
    sin_cos = @. sincos_mod(imag_mod(char_exp_v))
    return (@. w * last(sin_cos)), (@. w * first(sin_cos))
end
function lewis_coefficients(v, char_exp_v, ::BinaryPayoff)
    w = @. exp(real_mod(char_exp_v)) / (1 // 4 + v^2)
    sin_cos = @. sincos_mod(imag_mod(char_exp_v))
    return (@. w * (last(sin_cos) / 2 + v * first(sin_cos))), (@. w * (first(sin_cos) / 2 - v * last(sin_cos)))
end

# The nodes are equally spaced, v_j = v_0 + j h, so the sum over them is exp(c/2) Re[exp(i v_0 c) Σ_j C_j exp(i j h c)]:
# on the CPU each strike of a smile sums it with the Clenshaw recurrence, one sincos per strike instead of an exp and a cos per
# node and strike (a single option keeps the broadcast, as cheap). At the money h c is close to 0, hence Reinsch's form.
function lw_integrate(char_exp_v_r, abstractPayoff::SmilePayoff, df, dT, dx_adj, S0, rT_corr, ::FinancialMonteCarlo.SerialMode)
    corr_adj = @. log(S0 / abstractPayoff.K) + rT_corr + dT
    v = char_exp_v_r.v_im_adj
    c_re_unweighted, c_im_unweighted = lewis_coefficients(v, char_exp_v_r.char_exp_v, abstractPayoff)
    c_re = c_re_unweighted .* char_exp_v_r.weights
    c_im = c_im_unweighted .* char_exp_v_r.weights
    v_0 = first(v)
    h = (last(v) - v_0) / (length(v) - 1)
    total_sum_lewis = map(c -> lewis_clenshaw(c_re, c_im, v_0, h, c), corr_adj)
    return convert_integral_result_to_price_interf(dx_adj, total_sum_lewis, S0, dT, df, abstractPayoff)
end
function lewis_clenshaw(c_re, c_im, v_0, h, c)
    S_re, S_im = clenshaw_exp_sum(c_re, c_im, h * c)
    s, co = sincos_mod(v_0 * c)
    return exp(c / 2) * (co * S_re - s * S_im)
end

"""
    pricer(mcProcess, underlying, zero_rate, method, payoff)

Prices `payoff` under `mcProcess` with the Fourier method `method`
(`LewisMethod`, `CosMethod`, `CarrMadanLewisMethod`, `CarrMadanMethod`, `CosDensityMethod` or `FFTDensityInverter`).

# Arguments
- `mcProcess`: a `FinancialMonteCarlo.BaseProcess`, e.g. `BlackScholesProcess(σ)`.
- `underlying`: the `Underlying(S0, d)` (spot and dividends) priced under it.
- `zero_rate`: a `FinancialMonteCarlo.AbstractZeroRateCurve`, e.g. `ZeroRate(r)`.
- `method`: the configuration of the numerical method.
- `payoff`: an `EuropeanOption` or a `BinaryEuropeanOption`, a smile (`EuropeanOptionSmile`,
  `BinaryEuropeanOptionSmile`, several strikes sharing the same maturity). Options with different maturities are
  priced by broadcasting, `pricer.(mcProcess, underlying, zero_rate, method, payoffs)`.

The method carries the execution mode (the `mode` keyword of its constructor): in `FinancialMonteCarlo.CudaMode()` the
strikes of a smile are moved to the GPU, wherever they are, and its prices stay there.

Returns a price, or a vector of prices for smiles.
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, und::FinancialMonteCarlo.AbstractUnderlying, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FinancialFFT.LewisMethod, abstractPayoff_::FourierPayoff)
    mode = FinancialMonteCarlo.execution_mode(method)
    abstractPayoff = FinancialMonteCarlo.to_mode(abstractPayoff_, mode)
    T = abstractPayoff.T
    A_grid, N = ChainRulesCore.@ignore_derivatives quadrature_grid(method.grid, method, mcProcess, und, zero_rate, abstractPayoff)
    S0 = und.S0
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(und), T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    FT = float_type(S0, rT, dT, corr)
    A = to_float_type(FT, A_grid)
    # Quadrature: rectangle rule on [-A, A], every node with weight 1 (the trapezoidal rule but for the half weights of the
    # end nodes ±A, where the integrand is negligible); for an integrand analytic in a strip it converges geometrically in N,
    # until the truncation at ±A.
    # the N nodes (2j - (N - 1)) A / (N - 1), j = 0, …, N - 1, are symmetric: only the n_half of them in [0, A] are evaluated
    n_half = cld(N, 2)
    v, weights = ChainRulesCore.@ignore_derivatives begin
        j = (N-n_half):(N-1)
        adapt_array(collect(@. (2 * j - (N - 1)) * (A / (N - 1))), mode), PatternVector(n_half, InitialValuePattern(Int8(isodd(N) ? 1 : 2), Int8(2)))
    end
    # a float one_half: a Rational added to a complex number in a lazy argument slows the vectorised exponent down
    one_half = one(FT) / 2
    char_exp_v = @. FinancialFFT.characteristic_exponent_i(complex(one_half, v), mcProcess, T)
    lw_result = LewisMethodResult(char_exp_v, v, weights)
    dx_adj = A / (N - 1)
    df = exp(-rT)
    rT_corr = rT - corr
    return lw_integrate(lw_result, abstractPayoff, df, dT, dx_adj, S0, rT_corr, mode)
end
