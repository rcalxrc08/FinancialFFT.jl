"""
    DensityInverter(N, xmax = NaN)

Recovers the density of the log-moneyness `log(S_T / K)` by inverting the characteristic function with a
single FFT on `2^N` points spanning `[-xmax, xmax]`, then integrates the payoff against it.

# Arguments
- `N`: base-2 logarithm of the number of grid points (`N > 2`).
- `xmax`: half-width of the log-moneyness grid. It can be omitted only for `BlackScholesProcess`,
  where it defaults to `|drift| + 8σ√T`; for every other process it must be provided.
"""
struct DensityInverter{num_1 <: Integer, num <: Number} <: AbstractIntegralMethod
    N::num_1
    xmax::num
    function DensityInverter(N::num_1, xmax::num = NaN16) where {num_1 <: Integer, num <: Number}
        ChainRulesCore.ignore_derivatives() do
            if (!isnan(xmax))
                @assert(xmax > 0.0, "A must be positive")
            end
            @assert(N > 2, "N must be greater than 2")
        end
        return new{num_1, num}(N, xmax)
    end
end
using FFTW

function density_y(mcProcess, T, r, N, A, St, K, mode)
    n = ChainRulesCore.@ignore_derivatives 2^N
    vec_lin = ChainRulesCore.@ignore_derivatives 0:(n-1)
    vec = ChainRulesCore.@ignore_derivatives adapt_array(collect(-div(n, 2) .+ vec_lin), mode)            # Indices
    drift_rd = FinancialMonteCarlo.integral(r.r - mcProcess.underlying.d, T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    one_minus_one = ChainRulesCore.@ignore_derivatives PatternVector(n, EvenOddPattern(Int8(1), Int8(-1)))
    drift_rf = drift_rd + log(St) - log(K) - corr
    dt = pi / A # Step size, frequency space
    dt_im = dt * im
    drift_dt_im = drift_rf * dt_im
    Y_r = fft(@. exp(characteristic_exponent_i(vec * dt_im, mcProcess, T) + drift_dt_im * vec) * one_minus_one)
    twice_A = 2 * A
    eps_mod = ChainRulesCore.@ignore_derivatives typeof(drift_rf)(eps(zero(typeof(drift_rf))))
    density_vals_adj = @. max(abs(FinancialFFT.real_mod(Y_r)) / twice_A, eps_mod)
    ChainRulesCore.@ignore_derivatives @assert !isnan(sum(density_vals_adj) * twice_A / n) "Returned result is NaN"
    return density_vals_adj #/ almost_one
end

function density_x(N, A, mode)
    n = ChainRulesCore.@ignore_derivatives 2^N
    vec_lin = ChainRulesCore.@ignore_derivatives 0:(n-1)
    x_vec = ChainRulesCore.@ignore_derivatives collect(-1 .+ vec_lin * 2 / n)
    x = @. A * x_vec         # Grid, for the density
    return adapt_array(x, mode)
end

function adjust_price_adimensional(p, euPayoff::EuropeanOption)
    K = euPayoff.K
    return K * p
end
adjust_price_adimensional(V, ::BinaryEuropeanOption) = V

function compute_xmax(mcProcess::BlackScholesProcess, r, opt, method::DensityInverter)
    xmax = method.xmax
    S0 = mcProcess.underlying.S0
    d = mcProcess.underlying.d
    sigma = mcProcess.σ
    T = opt.T
    K = opt.K
    r_d = FinancialMonteCarlo.integral(r.r, T) - FinancialMonteCarlo.integral(d, T)
    mu = abs(log(S0) - log(K) + r_d - sigma^2 / 2 * T)
    sig = sigma * sqrt(T)
    # both branches promoted to a common type: the default xmax is a NaN16, and a Union return type breaks Mooncake
    xmax_default, xmax_user = promote(mu + 8 * sig, xmax)
    return isnan(xmax) ? xmax_default : xmax_user
end

function compute_xmax(::FinancialMonteCarlo.BaseProcess, _, _, method::DensityInverter)
    xmax = method.xmax
    @assert !isnan(xmax) "For generic processes xmax must be populated."
    return xmax
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::FinancialFFT.DensityInverter, opt, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    xmax = ChainRulesCore.@ignore_derivatives ignore_derivatives_mod(compute_xmax(mcProcess, r, opt, method))
    N = method.N
    T = opt.T
    f_x = density_y(mcProcess, T, r, N, xmax, mcProcess.underlying.S0, opt.K, mode)
    # x = ChainRulesCore.@ignore_derivatives exp.(density_x(N, xmax, mode))
    p = ChainRulesCore.@ignore_derivatives @. FinancialMonteCarlo.payout_untyped(exp($density_x(N, xmax, mode)), opt)
    n = ChainRulesCore.@ignore_derivatives 2^N
    df = exp(-FinancialMonteCarlo.integral(r.r, T))
    dx = 2 * xmax / n
    price = *(sum(p .* f_x), df, dx)
    return adjust_price_adimensional(price, opt)
end