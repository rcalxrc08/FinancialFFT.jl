# The parameters A and Npow of the Carr-Madan methods from a price tolerance ε (issue #41). Three errors, for the
# log-strike grid [-b, b) with step dk = 2π / A and b = π N / A:
# - interpolation and truncation: the FFT samples (1/π) Re ∫₀^A exp(-ivk) ψ(v) dv; six-point Lagrange interpolation errs on
#   exp(-ivk) by at most min(L₆ (v dk)⁶, 1 + Λ₆), and the truncation drops the v > A, so both are bounded by
#   E(A) = scale / π ∫₀^∞ |ψ(v)| w(v dk) dv, with w = 1 beyond A. E decreases with A: A_min solves E(A) = ε / 2.
# - aliasing: by Poisson's formula the trapezoidal rule adds the copies of the transformed function at the multiples of
#   2b, and the Simpson weights, the trapezoidal ones times 1 - (-1)^j / 3, the copies at the odd multiples of b with weight
#   -1/3. The copies at ±b give an error (1/3) exp(γ s) G(k + s), s = ±b, with G a price (call, binary, or the forward
#   minus the call) and γ its damping in the transform: each side gets ε / 4, from a closed form or from a Chernoff bound
#   on G, which gives b_min.
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
    s_opt = chernoff_optimum(exponent, log_epss, one(FT) / 8, FT(50), max(FT(1e-10), 8 * eps(FT)))
    return compute_ts(s_opt, exponent, log_epss)
end

# Quadrature nodes of E(A): trapezoidal rule in log v, on [1e-3, 1e6].
fft_error_nodes(FT) = exp.(range(log(one(FT) / 1000), log(FT(10)^6), length = 2048))

# The smallest A with E(A) <= ε, by bisection in log A; nothing if not reached at A = 1e6.
function interpolation_frequency_bound(modulus, scale::FT, ε::FT) where {FT}
    u = fft_error_nodes(FT)
    du = log(u[2]) - log(u[1])
    m = @. scale * du * u * modulus(u) * invπ
    L6, cap = FT(lagrange_error_constant), 1 + FT(lagrange_lebesgue_constant)
    weight(θ) = θ < twoπ ? min(L6 * θ^6, cap) : one(θ)
    E(A) = sum(m[i] * weight(twoπ * u[i] / A) for i in eachindex(u))
    lo, hi = one(FT), FT(10)^6
    E(hi) > ε && return nothing
    # 30 halvings of [0, log 1e6] give A to a relative 1e-8
    for _ = 1:30
        mid = sqrt(lo * hi)
        E(mid) > ε ? (lo = mid) : (hi = mid)
    end
    return hi
end

function fft_grid_parameters(A_min, b_min, Npow_max, ε)
    isnothing(A_min) && throw(ArgumentError("a tolerance of $ε needs A > 1e6: the characteristic function decays too slowly"))
    Npow = max(3, ceil(Int, log2(A_min * b_min * invπ)))
    Npow > Npow_max && throw(ArgumentError("a tolerance of $ε needs 2^$Npow nodes, more than 2^$Npow_max"))
    # not pi * 2^Npow: an irrational times an integer is a Float64
    return 2^Npow * (pi / b_min), Npow
end

# Strikes, maturity and market quantities of a payoff, without derivatives, and the tolerance, in the float type FT.
function fft_error_setup(model, zero_rate, opt, ε)
    T = v_value_mod(opt.T)
    S0 = v_value_mod(model.underlying.S0)
    rT = v_value_mod(FinancialMonteCarlo.integral(zero_rate.r, T))
    dT = v_value_mod(FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(model), T))
    FT = float_type(S0, rT, dT, T)
    K = to_float_type.(FT, v_value_mod.(opt.K))
    corr = to_float_type(FT, v_value_mod(FinancialFFT.characteristic_exponent_i(1, model, T)))
    return FT, T, to_float_type(FT, S0), to_float_type(FT, rT), to_float_type(FT, dT), minimum(K), maximum(K), corr, to_float_type(FT, ε)
end

"""
    CarrMadanMethod(process, zero_rate, payoff, ε; α = 0.75, Npow_max = 23)

`CarrMadanMethod(A, Npow, α)` whose error on the prices of `payoff` (an option or a smile, vanilla or binary) is below `ε`:
`A` and `Npow` come from bounds on the interpolation, truncation and aliasing errors (issue #41), computed in the float
type of the model, the rates and the payoff, which is the type of `A`. The bounds do not cover rounding: `ε` must be above
the rounding floor of that type. The parameters are constants for differentiation; throws an `ArgumentError` when more
than `2^Npow_max` nodes are needed.
"""
function CarrMadanMethod(model::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, opt::FourierPayoff, ε::Real; α::Real = 0.75, Npow_max::Integer = 23)
    A, Npow = ChainRulesCore.@ignore_derivatives begin
        FT, T, S0, rT, dT, K_min, _, corr, ε = fft_error_setup(model, zero_rate, opt, ε)
        drift = rT - dT
        F0 = S0 * exp(-dT)
        α_d = to_float_type(FT, v_value_mod(carr_madan_damping(model, T, to_float_type(FT, α))))
        k_min = log(K_min / S0)
        # |α + iv|² for a vanilla option, 1 for a binary (carr_madan_factor is identity or one)
        den = carr_madan_factor(opt)
        modulus(v) = v_value_mod(exp(real_mod(FinancialFFT.characteristic_exponent_i(v * im + 1 + α_d, model, T)) - (1 + α_d) * corr) / sqrt(den(α_d^2 + v^2) * ((1 + α_d)^2 + v^2)))
        # the price is F0 exp(-α (k - drift)) times the damped call, divided by K for a binary
        scale, P, c, γ = opt isa BinaryPayoff ? (F0 * exp(-α_d * (k_min - drift)) / K_min, exp(-rT), zero(FT), 1 + α_d) : (F0 * exp(-α_d * (k_min - drift)), F0, one(FT), α_d)
        log_epss = log(3 * ε / 4 / P)
        # copy at -b: G <= P; copy at +b: Chernoff
        b_min = max(-log_epss / γ, aliasing_chernoff_bound(model, T, c, γ, k_min, drift, corr, log_epss))
        fft_grid_parameters(interpolation_frequency_bound(modulus, scale, ε / 2), b_min, Npow_max, ε)
    end
    return CarrMadanMethod(A, Npow, α)
end

"""
    CarrMadanLewisMethod(process, zero_rate, payoff, ε; Npow_max = 23)

`CarrMadanLewisMethod(A, Npow)` whose error on the prices of `payoff` (an option or a smile, vanilla or binary) is below
`ε`, as for [`CarrMadanMethod`](@ref).
"""
function CarrMadanLewisMethod(model::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, opt::FourierPayoff, ε::Real; Npow_max::Integer = 23)
    A, Npow = ChainRulesCore.@ignore_derivatives begin
        FT, T, S0, rT, dT, K_min, K_max, corr, ε = fft_error_setup(model, zero_rate, opt, ε)
        drift = rT - dT
        F0 = S0 * exp(-dT)
        correction = drift - corr
        den = lewis_denominator(opt)
        # |den(x)| = den(|x|) for both abs2 and identity
        modulus(v) = v_value_mod(exp(real_mod(correction * (1 // 2 + v * im) + FinancialFFT.characteristic_exponent_i(1 // 2 + v * im, model, T))) / den(sqrt(1 // 4 + v^2)))
        if opt isa BinaryPayoff
            # the integral is exp(k / 2) times the binary: copy at -b below exp(-rT), Chernoff at +b
            scale = sqrt(S0 / K_min) * exp(-rT)
            log_epss = log(3 * ε / 4 / exp(-rT))
            b_min = max(-2 * log_epss, aliasing_chernoff_bound(model, T, zero(FT), one(FT) / 2, log(K_min / S0), drift, corr, log_epss))
        else
            # the integral is exp(-k / 2) (F0 - C) / S0, with F0 - C below F0 and K exp(-rT): closed forms on both sides
            scale = sqrt(S0 * K_max) * exp(-rT)
            b_min = 2 * log(4 * max(F0, K_max * exp(-rT)) / (3 * ε))
        end
        fft_grid_parameters(interpolation_frequency_bound(modulus, scale, ε / 2), b_min, Npow_max, ε)
    end
    return CarrMadanLewisMethod(A, Npow)
end
