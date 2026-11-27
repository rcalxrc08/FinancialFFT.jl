# The number of terms of CosMethod (and the tail probability of its range): given, or chosen at pricing time from a price
# tolerance (#48).
struct CosTerms{num_1 <: Integer}
    N::num_1
    function CosTerms(N::num_1) where {num_1 <: Integer}
        check_nodes(N)
        return new{num_1}(N)
    end
end
struct CosTolerance{num <: Real, num_1 <: Integer}
    ε::num
    N_max::num_1
    function CosTolerance(ε::num, N_max::num_1) where {num <: Real, num_1 <: Integer}
        check_quadrature_tolerance(ε, N_max)
        return new{num, num_1}(ε, N_max)
    end
end

"""
    CosMethod(N; mode = SerialMode())
    CosMethod(ε; N_max = 2^20, mode = SerialMode())

Fourier-cosine expansion method of Fang & Oosterlee (2008). The density of the log-price is expanded on
a truncation interval `[a, b]` with the cosine terms `k = 0, …, N`; the interval is computed from a
Chernoff bound on the tails of the process (see [Truncation range](@ref)), so no range parameter is needed.
The expansion is on the put, and calls follow from parity: a call needs no information on the right tail of the process.

With a tolerance `ε` (a float), the number of terms (and, for a very small `ε`, the tail probability of the range) is
chosen at pricing time, for the model, the maturity and the strikes, so that the price error stays below `ε`; both
come from the characteristic function `φ`:
- the range: the mass outside it, weighted by the cosine coefficients of the put, gets `ε / 2`; the tail probability
  is never above the `1e-14` of the fixed form, which the derivatives need (see [Truncation range](@ref)).
- `N` (series truncation): the terms beyond `N` are below `|φ(u_k)|` times an envelope of the cosine coefficients of
  the put (decaying like `1 / u²`, like `1 / u` for a binary), at the frequencies `u_k = k π / (b - a)`. Their sum is
  estimated by the integral of this product beyond `u_N` (at the parameters of the model, an estimate rather than a
  rigorous bound) and gets `ε / 2`. A wide range (heavy tails, such as Kou with rare up jumps) or a slowly decaying `φ`
  (a short maturity, a small variance) asks for more terms.
The choice costs about one price; for repeated pricing resolve it once with `FinancialFFT.cos_range_and_terms` and
pass `CosMethod(N)`. The bounds do not cover the rounding: `ε` must stay above the floor of the formula.

# Arguments
- `N`: number of cosine terms (`N > 2`).
- `ε`: the price tolerance, an `AbstractFloat` (`ε > 0`): an integer argument is never read as a tolerance.
- `N_max` (keyword): the largest number of terms the tolerance may ask for; more throws an `ArgumentError`.
- `mode` (keyword): the execution mode, `FinancialMonteCarlo.SerialMode()` (default) or `FinancialMonteCarlo.CudaMode()` with CUDA.jl loaded; `FinancialMonteCarlo.with_mode(method, mode)` changes it.
"""
struct CosMethod{G <: Union{CosTerms, CosTolerance}, modeType <: FinancialMonteCarlo.AbstractMode} <: FinancialFFT.AbstractIntegralMethod
    grid::G
    mode::modeType
    function CosMethod(grid::G, mode::modeType) where {G <: Union{CosTerms, CosTolerance}, modeType <: FinancialMonteCarlo.AbstractMode}
        return new{G, modeType}(grid, mode)
    end
end
CosMethod(N::Integer, mode::FinancialMonteCarlo.AbstractMode) = CosMethod(CosTerms(N), mode)
CosMethod(N::Integer; mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = CosMethod(CosTerms(N), mode)
CosMethod(ε::AbstractFloat; N_max::Integer = 2^20, mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = CosMethod(CosTolerance(ε, N_max), mode)
export CosMethod
# The expansion is on the put, whose payoff is bounded. The coefficients of a call grow like exp(b): they lose accuracy
# to rounding when the range is wide, and the heavy right tail they need is beyond it when it is not. Parity needs neither.
# Payoff of the put against cos(u (y - a)) for y = log(S_T / K) in [a, h], with k = 0 halved: adj flags that term.
@muladd function cos_binary_put_integral(u, adj, a, h)
    return (1 + !adj) * (adj * (h - a) + sin(u * (h - a)) / (adj + u))
end

function cos_put_integral(u, adj, a, h)
    sin_u, cos_u = sincos(u * (h - a))
    return cos_binary_put_integral(u, adj, a, h) - (1 + !adj) * (exp(h) * (cos_u + u * sin_u) - exp(a)) / (1 + u^2)
end

cos_payoff_integral(::VanillaPayoff) = cos_put_integral
cos_payoff_integral(::BinaryPayoff) = cos_binary_put_integral

# Σ_k coefficient_k cos(u_k x + phase_k) for each x (a number or a vector, one per strike).
# On the CPU it is the real part of Σ_k C_k exp(i k θ), with C_k = coefficient_k exp(i phase_k) and θ = π x / (b - a): the
# Clenshaw recurrence costs a sincos per strike instead of a cos per term and strike, and the phases (which
# do not depend on the strike) are folded into C_k once. Other modes (GPU) cannot index the coefficients, and keep one cos per term.
function cos_phased_series(coefficient, phase, u, x, ::FinancialMonteCarlo.SerialMode)
    sin_cos = sincos_mod.(phase)
    c_re = @. coefficient * last(sin_cos)
    c_im = @. coefficient * first(sin_cos)
    θ = x .* (last(u) / (length(u) - 1))
    return map(θ_j -> cos_clenshaw(c_re, c_im, θ_j), θ)
end
cos_phased_series(coefficient, phase, u, x, ::FinancialMonteCarlo.AbstractMode) = phased_cos_sum(coefficient, phase, u, x)

# Re Σ_k (c_re_k + i c_im_k) exp(i k θ) = Σ_k c_re_k cos(k θ) - c_im_k sin(k θ).
cos_clenshaw(c_re, c_im, θ) = clenshaw_exp_sum(c_re, c_im, θ, Val(true))

# The range [a, b] of log(S_T / S0) and the number of terms N of a CosMethod, for this model and payoff (constants for
# differentiation: computed on the values). cos_range_and_terms(method.grid, model, und, zero_rate, payoff).
function cos_range_and_terms(grid::CosTerms, model, und, zero_rate, opt)
    T = v_value_mod(opt.T)
    rT = v_value_mod(FinancialMonteCarlo.integral(zero_rate.r, T))
    dT = v_value_mod(FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(und), T))
    a, b = compute_chernorff_limits(model, rT, dT, T)
    return v_value_mod(a), v_value_mod(b), grid.N
end

# Envelope of the cosine coefficients of the put on y = log(S_T / K) in [a, h] (cos_put_integral, cos_binary_put_integral,
# k >= 1, the factor 2 included), at the frequency u, with L = h - a:
# - vanilla: 2 [sin(uL) (1/u - e^h u / (1 + u²)) - (e^h cos(uL) - e^a) / (1 + u²)], below
#   2 [min(1, uL) |1/u - e^h u / (1 + u²)| + (e^h + e^a) / (1 + u²)], decaying like 1/u²;
# - binary: 2 sin(uL) / u, below 2 min(L, 1/u).
struct CosPayoffEnvelope{V, FT}
    a::FT
    h::FT
end
cos_payoff_envelope(::VanillaPayoff, a, h) = CosPayoffEnvelope{VanillaPayoff, typeof(a)}(a, h)
cos_payoff_envelope(::BinaryPayoff, a, h) = CosPayoffEnvelope{BinaryPayoff, typeof(a)}(a, h)
(e::CosPayoffEnvelope{VanillaPayoff})(u) = 2 * (min(one(u), u * (e.h - e.a)) * abs(inv(u) - exp(e.h) * u / (1 + u^2)) + (exp(e.h) + exp(e.a)) / (1 + u^2))
(e::CosPayoffEnvelope{BinaryPayoff})(u) = 2 * min(e.h - e.a, inv(u))

# CosMethod(ε): the price is P Σ_k c_k cos(…) with P = K e^{-rT} (vanilla) or e^{-rT} (binary) and |c_k| below
# envelope(u_k) |φ(u_k)| / (b - a), u_k = k π / (b - a), so that Σ_{k > N} ≈ (1/π) ∫_{u_N}^∞ envelope |φ| du, the
# integral of the Lewis and Carr-Madan bounds (frequency_error_terms, fft_parameters.jl). Two errors, ε / 2 each:
# - the range: the coefficients use φ, the transform of the whole density, so the mass outside [a, b] enters every term,
#   weighted by C = Σ_k |payoff_k| / (b - a) (computed with the same envelope); the put payoff is below P there too.
#   A tail probability ε / (8 P C) per side covers both; never above the 1e-14 of the fixed form.
# - the series: the smallest u* whose tail integral is below ε / 2, and N = ceil(u* (b - a) / π).
# The range of log(S_T / S0) for a tail probability, its union on y = log(S_T / K) over the strikes, and the envelope of
# the payoff coefficients on it.
function cos_range_on_strikes(model, T, tail, drift, S0, K_min, K_max, opt)
    FT = typeof(tail)
    lims = compute_chernoff_extrema(model, T, tail, drift, one(FT), sqrt(eps(FT)))
    a, b = lims[1] + log(S0 / K_max), lims[2] + log(S0 / K_min)
    return lims, b - a, cos_payoff_envelope(opt, a, max(a, min(zero(FT), b)))
end
# |term| of the series at the frequency u, up to 1 / (b - a): a callable, not a closure over a reassigned envelope (a
# boxed variable makes the pricer not inferred)
struct CosTermModulus{E, M, TT}
    envelope::E
    model::M
    T::TT
end
(m::CosTermModulus)(u) = m.envelope(u) * v_value_mod(exp(real_mod(FinancialFFT.characteristic_exponent_i(u * im, m.model, m.T))))

function cos_range_and_terms(grid::CosTolerance, model, und, zero_rate, opt)
    FT, T, S0, rT, dT, K_min, K_max, _, ε = fft_error_setup(model, und, zero_rate, opt, grid.ε)
    P = opt isa BinaryPayoff ? exp(-rT) : K_max * exp(-rT)
    drift = rT - dT
    floor_tail = scaled_tolerance(FT, 1e-14)
    lims, bma, envelope = cos_range_on_strikes(model, T, floor_tail, drift, S0, K_min, K_max, opt)
    C = (envelope.h - envelope.a) / bma + sum(last(frequency_error_terms(envelope, one(FT))))
    # a constant, as the floor: a dual tail reaches the bracketing of the Chernoff optimum (ForwardDiff, large strikes)
    tail = ignore_derivatives_mod(ε / (8 * P * C))
    if tail < floor_tail
        lims, bma, envelope = cos_range_on_strikes(model, T, tail, drift, S0, K_min, K_max, opt)
    end
    u_star = ignore_derivatives_mod(truncation_frequency_bound(CosTermModulus(envelope, model, T), P, ε / 2, ε))
    N = max(8, ceil(Int, u_star * bma * invπ))
    N > grid.N_max && throw(ArgumentError("a tolerance of $(grid.ε) needs $N terms, more than N_max = $(grid.N_max)"))
    return ignore_derivatives_mod(lims[1]), ignore_derivatives_mod(lims[2]), N
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, und::FinancialMonteCarlo.AbstractUnderlying, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, opt_::FourierPayoff)
    mode = FinancialMonteCarlo.execution_mode(method)
    opt = FinancialMonteCarlo.to_mode(opt_, mode)
    T = opt.T
    S0 = und.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(und), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    K = opt.K
    # The limits are for log(S_T / S0): on y = log(S_T / K) they move with the strike, and the range covers all of them.
    # Sharing it among the strikes shares the payoff coefficients: only the phase below depends on the strike.
    a_S0, b_S0, N = ChainRulesCore.@ignore_derivatives cos_range_and_terms(method.grid, mcProcess, und, r, opt)
    a, b = ChainRulesCore.@ignore_derivatives begin
        K_min, K_max = extrema(K)
        v_value_mod(a_S0 + log(S0 / K_max)), v_value_mod(b_S0 + log(S0 / K_min))
    end
    bma = b - a
    # A vector, not a float range: Enzyme, which ignores @ignore_derivatives, has no derivative for the TwicePrecision of
    # Base.floatrange (EnzymeAD/Enzyme.jl#4016, FinancialCalibrator.jl#41), as in carrmadan.jl. k * du is rounded once.
    du = pi / bma
    u = ChainRulesCore.@ignore_derivatives FinancialFFT.adapt_array([k * du for k = 0:N], mode)
    payoff = ChainRulesCore.@ignore_derivatives begin
        adj = PatternVector(length(u), InitialValuePattern(true, false))
        cos_payoff_integral(opt).(u, adj, a, max(a, min(0, b)))
    end
    driftT = rT - FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    v_char_exp = @. FinancialFFT.characteristic_exponent_i(u * im, mcProcess, T)
    coefficient = @. payoff * exp(FinancialFFT.real_mod(v_char_exp)) / bma
    phase = @. FinancialFFT.imag_mod(v_char_exp)
    S0_adj = S0 * exp(-dT)
    x = @. log(S0_adj / K) + driftT - a
    df = exp(-rT)
    put_price = adjust_price_adimensional(cos_phased_series(coefficient, phase, u, x, mode), opt) .* df
    return put_to_option(put_price, S0_adj, df, opt)
end
