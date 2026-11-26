# The parameters A and Npow of the Carr-Madan methods from a price tolerance ε (issue #41). Three errors, for the
# log-strike grid [-b, b) with step dk = 2π / A and b = π N / A:
# - interpolation and truncation: the FFT samples (1/π) Re ∫₀^A exp(-ivk) ψ(v) dv; six-point Lagrange interpolation errs on
#   exp(-ivk) by at most min(L₆ (v dk)⁶, 1 + Λ₆), and the truncation drops the v > A, so both are bounded by
#   E(A) = scale / π ∫₀^∞ |ψ(v)| w(v dk) dv, with w = 1 beyond A. E decreases with A: A_min solves E(A) = ε / 2.
# - aliasing: by Poisson's formula the trapezoidal rule (#104) adds the copies of the transformed function at the multiples
#   of the period 2b. The copies at ±2b give an error exp(γ s) G(k + s), s = ±2b, with G a price (call, binary, or the
#   forward minus the call) and γ its damping in the transform; weight 2 on them covers the further copies. Each side gets
#   ε / 4, from a closed form or from a Chernoff bound on G: the distance D_min, and b_min = D_min / 2. (Simpson's weights,
#   used before #104, added copies at ±b with weight 1/3, so b had to reach the distance itself: one more power of two.)
# Then N = 2^Npow >= A_min b_min / π, and A = π N / b_min: the spare nodes go to A, which reduces the first error.

# Every quantity is in the float type FT of the model, the rates and the payoff (float_type), so that the parameters of a
# Float32 or BigFloat model come from computations in its type.

# six-point stencil on the central interval, at t = 1/2: max |Π (t - tᵢ)| / 6! = (225 / 64) / 720 and the Lebesgue constant
# Σ |ℓᵢ(1/2)|, as rationals (converted to FT once per call)
const lagrange_error_constant = 5 // 1024
const lagrange_lebesgue_constant = 89 // 64

# Chernoff exponent of the copy at +b, in s = p - c - γ: G(k') <= P exp(Λ(p) - (p - c)(k' - drift)) with
# Λ(p) = Ψ_T(p) - p Ψ_T(1), so the copy is below ε for b >= t(s) = (Λ(s + c + γ) - (s + γ)(k_min - drift) - log(3ε / P)) / s.
struct AliasingExponent{M, TT, F}
    model::M
    T::TT
    shift::F # c + γ
    γ::F
    slope::F # k_min - drift
    corr::F  # Ψ_T(1)
end
(e::AliasingExponent)(s) = FinancialFFT.characteristic_exponent_i(s + e.shift, e.model, e.T) - (s + e.shift) * e.corr - (s + e.γ) * e.slope
exponential_moment_bound(e::AliasingExponent) = exponential_moment_bound(e.model, 1, e.T) - e.shift
complex_step(s, e::AliasingExponent) = complex_step(s, e.model)

# c, γ, k_min, drift, corr and log_epss of type FT
function aliasing_chernoff_bound(model, T, c, γ, k_min, drift, corr, log_epss)
    FT = typeof(log_epss)
    exponent = AliasingExponent(model, T, c + γ, γ, k_min - drift, corr)
    s_opt = chernoff_optimum(exponent, log_epss, one(FT) / 8, sqrt(eps(FT)))
    return compute_ts(s_opt, exponent, log_epss)
end

# Quadrature nodes of E(A): trapezoidal rule in log v (order du², weights from PatternVectors), on [1e-3, 1e6].
fft_error_nodes(FT) = exp.(range(log(one(FT) / 1000), log(FT(10)^6), length = 2048))

# The terms of E(A) = scale / π ∫ |ψ(v)| w(v) dv on the nodes u of fft_error_nodes: trapezoidal rule in log v, weights
# du / 2 times 1, 2, …, 2, 1 (PatternVectors), and the Jacobian u of v = exp(log v).
function frequency_error_terms(modulus, scale::FT) where {FT}
    u = fft_error_nodes(FT)
    du = log(u[2]) - log(u[1])
    trapezoid = PatternVector(length(u), PaddedFillPattern(Int8(1), Int8(2), Int8(1)))
    return u, @. scale * (du / 2) * trapezoid * u * modulus(u) * invπ
end
frequency_bound_error(tolerance) = ArgumentError("a tolerance of $tolerance needs A > 1e6: the characteristic function decays too slowly")

# Carr-Madan methods: the smallest A with E(A) <= ε, where the weight is the six-point interpolation error below A and 1
# beyond it. E is continuous and decreasing in A, and A enters every term (through 2π v / A): bisection in log A, which
# needs no derivative. Throws if not reached at A = 1e6 (tolerance: the one asked for, for the message).
function interpolation_frequency_bound(modulus, scale::FT, ε::FT, tolerance) where {FT}
    u, m = frequency_error_terms(modulus, scale)
    L6, cap = FT(lagrange_error_constant), 1 + FT(lagrange_lebesgue_constant)
    weight(θ) = θ < twoπ ? min(L6 * θ^6, cap) : one(θ)
    E(A) = sum(m[i] * weight(twoπ * u[i] / A) for i in eachindex(u))
    lo, hi = one(FT), FT(10)^6
    E(hi) > ε && throw(frequency_bound_error(tolerance))
    # 30 halvings of [0, log 1e6] give A to a relative 1e-8
    for _ = 1:30
        mid = sqrt(lo * hi)
        E(mid) > ε ? (lo = mid) : (hi = mid)
    end
    return hi
end

# LewisMethod, which does not interpolate: the truncation only, E(A) = the sum of the terms at the nodes u >= A, a step
# function of A. Its smallest node whose tail sum is below ε comes from one reverse cumulative sum, without a search:
# exact at the resolution of the nodes (1% in A), and conservative (the node, not the step below it). At least 1.
function truncation_frequency_bound(modulus, scale::FT, ε::FT, tolerance) where {FT}
    u, m = frequency_error_terms(modulus, scale)
    tail = reverse(cumsum(reverse(m)))
    last(tail) > ε && throw(frequency_bound_error(tolerance))
    return max(one(FT), u[findfirst(<=(ε), tail)])
end

# A_min and b_min become constants for differentiation here: ReverseDiff tracks the Chernoff bound (it does not honor
# @ignore_derivatives) and would break the integer conversion of Npow, and Mooncake would differentiate A, adding the
# derivative of the price in A to the gradients.
function fft_grid_parameters(A_min, b_min, Npow_max, ε)
    A_min, b_min = ignore_derivatives_mod(A_min), ignore_derivatives_mod(b_min)
    Npow = max(3, ceil(Int, log2(A_min * b_min * invπ)))
    Npow > Npow_max && throw(ArgumentError("a tolerance of $ε needs 2^$Npow nodes, more than 2^$Npow_max"))
    # not pi * 2^Npow: an irrational times an integer is a Float64
    return 2^Npow * (pi / b_min), Npow
end

# Strikes, maturity and market quantities of a payoff, without derivatives, and the tolerance, in the float type FT.
function fft_error_setup(model, und, zero_rate, opt, ε)
    T = v_value_mod(opt.T)
    S0 = v_value_mod(und.S0)
    rT = v_value_mod(FinancialMonteCarlo.integral(zero_rate.r, T))
    dT = v_value_mod(FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(und), T))
    FT = float_type(S0, rT, dT, T)
    K = to_float_type.(FT, v_value_mod.(opt.K))
    corr = to_float_type(FT, v_value_mod(FinancialFFT.characteristic_exponent_i(1, model, T)))
    return FT, T, to_float_type(FT, S0), to_float_type(FT, rT), to_float_type(FT, dT), minimum(K), maximum(K), corr, to_float_type(FT, ε)
end

# The grid of CarrMadanMethod(ε): A and Npow for the model and the strikes of opt.
function fft_grid(grid::FFTTolerance, method::CarrMadanMethod, model, und, zero_rate, opt)
    FT, T, S0, rT, dT, K_min, _, corr, ε = fft_error_setup(model, und, zero_rate, opt, grid.ε)
    drift = rT - dT
    F0 = S0 * exp(-dT)
    α_d = to_float_type(FT, v_value_mod(carr_madan_damping(model, T, to_float_type(FT, method.α))))
    k_min = log(K_min / S0)
    # |α + iv|² for a vanilla option, 1 for a binary (carr_madan_factor is identity or one)
    den = carr_madan_factor(opt)
    modulus(v) = v_value_mod(exp(real_mod(FinancialFFT.characteristic_exponent_i(v * im + 1 + α_d, model, T)) - (1 + α_d) * corr) / sqrt(den(α_d^2 + v^2) * ((1 + α_d)^2 + v^2)))
    # the price is F0 exp(-α (k - drift)) times the damped call, divided by K for a binary
    scale, P, c, γ = opt isa BinaryPayoff ? (F0 * exp(-α_d * (k_min - drift)) / K_min, exp(-rT), zero(FT), 1 + α_d) : (F0 * exp(-α_d * (k_min - drift)), F0, one(FT), α_d)
    # weight 2 on the copies at ±2b
    log_epss = log(ε / 8 / P)
    # copy at -2b: G <= P; copy at +2b: Chernoff
    b_min = max(-log_epss / γ, aliasing_chernoff_bound(model, T, c, γ, k_min, drift, corr, log_epss)) / 2
    return fft_grid_parameters(interpolation_frequency_bound(modulus, scale, ε / 2, ε), b_min, grid.Npow_max, ε)
end

# The Lewis integrand on the line 1/2 + iv, shared by CarrMadanLewisMethod(ε) and LewisMethod(ε): the price is
# scale / π ∫₀^∞ of it (up to terms that do not depend on the quadrature), so its modulus bounds the truncation, and the
# quantities of the aliasing bounds.
function lewis_error_setup(model, und, zero_rate, opt, ε)
    FT, T, S0, rT, dT, K_min, K_max, corr, ε = fft_error_setup(model, und, zero_rate, opt, ε)
    drift = rT - dT
    correction = drift - corr
    den = lewis_denominator(opt)
    # |den(x)| = den(|x|) for both abs2 and identity
    modulus(v) = v_value_mod(exp(real_mod(correction * (1 // 2 + v * im) + FinancialFFT.characteristic_exponent_i(1 // 2 + v * im, model, T))) / den(sqrt(1 // 4 + v^2)))
    # the integral is exp(k / 2) times the binary, exp(-k / 2) (F0 - C) / S0 for a vanilla option
    scale = opt isa BinaryPayoff ? sqrt(S0 / K_min) * exp(-rT) : sqrt(S0 * K_max) * exp(-rT)
    return (; FT, T, S0, rT, F0 = S0 * exp(-dT), K_min, K_max, corr, drift, ε, modulus, scale)
end

# The smallest log-strike distance at which a copy of the transformed price function, with weight w, is below ε / 4.
# - binary: exp(k / 2) times the binary; the copy at -L is below w exp(-rT) exp(-L / 2), the one at +L from a Chernoff
#   bound on the binary.
# - vanilla: exp(-k / 2) (F0 - C) / S0, with F0 - C below F0 and K exp(-rT): closed forms on both sides, no model.
function lewis_aliasing_distance(setup, model, opt, w)
    (; FT, T, S0, rT, F0, K_min, K_max, corr, drift, ε) = setup
    if opt isa BinaryPayoff
        log_epss = log(ε / 4 / w / exp(-rT))
        return max(-2 * log_epss, aliasing_chernoff_bound(model, T, zero(FT), one(FT) / 2, log(K_min / S0), drift, corr, log_epss))
    end
    return 2 * log(4 * w * max(F0, K_max * exp(-rT)) / ε)
end

# The grid of CarrMadanLewisMethod(ε): trapezoidal copies at ±2b, weight 2 (covering the further ones).
function fft_grid(grid::FFTTolerance, ::CarrMadanLewisMethod, model, und, zero_rate, opt)
    setup = lewis_error_setup(model, und, zero_rate, opt, grid.ε)
    b_min = lewis_aliasing_distance(setup, model, opt, 2 * one(setup.FT)) / 2
    return fft_grid_parameters(interpolation_frequency_bound(setup.modulus, setup.scale, setup.ε / 2, setup.ε), b_min, grid.Npow_max, setup.ε)
end
