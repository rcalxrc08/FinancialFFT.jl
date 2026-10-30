"""
    CarrMadanMethod(A, Npow, α = 0.75)

Carr–Madan (1999) FFT method. The damped call transform is integrated with Simpson weights
on the frequency grid `v ∈ [0, A)` with `2^Npow` nodes, a single FFT returns call prices on a
log-strike grid, and prices at the requested strikes are obtained through cubic-spline interpolation.

# Arguments
- `A`: upper bound of the frequency integration domain (`A > 0`). The strike grid has step `2π / A`.
- `Npow`: base-2 logarithm of the number of FFT nodes (`2 < Npow < 24`).
- `α`: damping exponent (`α > 0`). It needs `E[S^(1 + α)]` to be finite, so it is capped halfway between 1 and the
  exponential moment bound of the model. The frequency step `A / 2^Npow` must be small with respect to the capped value.

The spline step is excluded from ChainRules-based differentiation, so Zygote gradients are not reliable for this method.
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

using DataInterpolations, FFTW;
# E[S^(1 + α)] must be finite: 1 + α stays at most halfway between 1 and the exponential moment bound of the upper tail.
carr_madan_damping(mcProcess, T, α) = min(α, (exponential_moment_bound(mcProcess, 1, v_value_mod(T)) - 1) / 2)
"""
    pricer(mcProcess, StrikeVec, zero_rate, T, method::Union{CarrMadanMethod, CarrMadanLewisMethod}[, mode])

Prices European **call** options with maturity `T` and strikes `StrikeVec` with a single FFT.

# Arguments
- `mcProcess`: a `FinancialMonteCarlo.BaseProcess`, e.g. `BlackScholesProcess(σ, Underlying(S0, d))`.
- `StrikeVec`: vector of strikes.
- `zero_rate`: a `FinancialMonteCarlo.AbstractZeroRateCurve`, e.g. `ZeroRate(r)`.
- `T`: maturity.
- `method`: a `CarrMadanMethod` or a `CarrMadanLewisMethod`.
- `mode`: `FinancialMonteCarlo.SerialMode()` (default) or a CUDA mode when CUDA.jl is loaded.

Returns a vector of call prices, one per strike.
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanMethod, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    A = method.A
    #* v-> compute integral as a summation
    dx = A / N
    real_vec = 0:(N-1)
    v = ChainRulesCore.@ignore_derivatives collect(real_vec * dx)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    one_adj = ChainRulesCore.@ignore_derivatives Int8(1)
    minus_one_adj = ChainRulesCore.@ignore_derivatives Int8(-1)
    one_minus_one = ChainRulesCore.@ignore_derivatives PatternVector(N, EvenOddPattern(one_adj, minus_one_adj))
    two_adj = ChainRulesCore.@ignore_derivatives Int8(2)
    four_adj = ChainRulesCore.@ignore_derivatives Int8(4)
    weights_ = ChainRulesCore.@ignore_derivatives PatternVector(N, PaddedEvenOddPattern(one_adj, four_adj, two_adj, one_adj))
    dk = 2 / A * pi
    b = pi * N / A
    drift_rd = rT - dT
    v_im = @. v * im
    alfa = ChainRulesCore.@ignore_derivatives carr_madan_damping(mcProcess, T, method.α)
    # transform of exp(α k) C(k): E[exp((1 + α + iv) X)] / ((α + iv) (1 + α + iv)), i.e. the exponent evaluated in 1 + α + iv
    complex_vec_z_k = @. one_minus_one * cis(drift_rd * v) * exp(FinancialFFT.characteristic_exponent_i(v_im + (1 + alfa), mcProcess, T) - (v_im + 1 + alfa) * corr) / ((v_im + alfa) * (alfa + 1 + v_im)) * dx * weights_ / 3
    y = fft(complex_vec_z_k)
    z_T = @. real_mod(y) / pi

    # not -b .+ dk * real_vec: a TwicePrecision range breaks Mooncake when it reuses the rule (no _copy method)
    ks = ChainRulesCore.@ignore_derivatives StepRangeLen(-b, dk, N)
    spline_cub = ChainRulesCore.@ignore_derivatives CubicSplineInterpolation(ks, z_T)
    k_interp = @. log(StrikeVec / S0)
    # map and not a broadcast: ReverseDiff tracks the coefficients of the spline, and cannot broadcast a function that
    # captures tracked values
    return @. S0 * exp(-dT - alfa * (k_interp - drift_rd)) * $map(spline_cub, k_interp)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanMethod, abstractPayoffs::Array{U}, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = abstractPayoffs[index_same_t]
        strikes = [opt.K for opt in payoffs]
        prices_call = pricer(mcProcess, strikes, zero_rate, T, method)
        dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
        prices_final = [call_to_put(prices_call[i], mcProcess.underlying.S0 * exp(dT), exp(-FinancialMonteCarlo.integral(zero_rate.r, T)), payoffs[i]) for i in eachindex(payoffs)]
        prices[index_same_t] = prices_final
    end

    return prices
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    res = first(pricer(mcProcess, [abstractPayoff.K], zero_rate, abstractPayoff.T, method))
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), abstractPayoff.T)
    return call_to_put(res, mcProcess.underlying.S0 * exp(dT), exp(-FinancialMonteCarlo.integral(zero_rate.r, abstractPayoff.T)), abstractPayoff)
end
