using FinancialMonteCarlo, MuladdMacro, ChainRulesCore

# The exponent of a Chernoff bound, s -> log E[exp(mul s X_T)]: P(mul (X_T - E) > t) <= exp(exponent(s) - s t) for every
# admissible s > 0. Other bounds (the aliasing of the Carr-Madan methods, fft_parameters.jl) define their own exponent,
# with complex_step and exponential_moment_bound methods.
struct TailExponent{M, TT, I}
    model::M
    T::TT
    mul::I
end
(e::TailExponent)(s) = FinancialFFT.characteristic_exponent_i(e.mul * s, e.model, e.T)
exponential_moment_bound(e::TailExponent) = exponential_moment_bound(e.model, e.mul, e.T)

# t(s) = (exponent(s) - log ε) / s: the tail beyond t(s) has probability at most ε.
compute_ts(s, exponent, log_epss) = @muladd inv(s) * (exponent(s) - log_epss)
compute_ts(s, model, T, log_epss, mul) = compute_ts(s, TailExponent(model, T, mul), log_epss)

complex_step(s, model) = eps(s)
# Large imaginary parts cancel out in the Heston exponent when γ is imaginary: a step of the order of eps is lost in the rounding.
complex_step(s, ::FinancialMonteCarlo.HestonProcess) = sqrt(eps(s))
complex_step(s, e::TailExponent) = complex_step(s, e.model)

# t'(s) = (g(s) - L) / s², with g(s) = s κ'(s) - κ(s) for the exponent κ and L = -log ε. For a convex κ, g'(s) = s κ''(s) > 0:
# g - L changes sign at most once, from negative to positive, at the only stationary point of t, its minimum. g comes from
# one complex step, κ(s + ih) = κ(s) + ih κ'(s) + O(h²): the real part is κ(s), the imaginary one over h is κ'(s).
function chernoff_g(s, exponent)
    h = complex_step(s, exponent)
    κ_s = exponent(@muladd s + im * h)
    κ = v_value_mod(FinancialFFT.real_mod(κ_s))
    dκ = v_value_mod(FinancialFFT.imag_mod(κ_s)) / h
    return @muladd s * dκ - κ
end

# The s with g(s) = L if g were Gaussian (g(s) = κ''(0) s² / 2, exact for Black Scholes), from g at one s: it scales like
# 1 / sqrt(T) for every Lévy model, and heavier tails have the root below it. Doubles s when g(s) <= 0 (an exponent with
# κ(0) > 0, the aliasing one of fft_parameters.jl).
gaussian_root_guess(s, g, L) = g > 0 ? s * sqrt(L / g) : 2 * s

# Brent's method (Brent 1973, ch. 4) for the root of f in [a, b], with f(a) and f(b) of opposite signs: inverse quadratic
# interpolation or secant steps, kept inside the bracket and replaced by bisection when they do not shrink it fast
# enough, down to a width xtol. Never evaluates f outside [a, b].
function brent_root(f, a, b, f_a, f_b, xtol, max_iterations = 100)
    c, f_c = a, f_a
    d = e = b - a
    for _ = 1:max_iterations
        if signbit(f_b) == signbit(f_c)
            c, f_c = a, f_a
            d = e = b - a
        end
        if abs(f_c) < abs(f_b)
            a, b, c = b, c, b
            f_a, f_b, f_c = f_b, f_c, f_b
        end
        tol = 2 * eps(abs(b)) + xtol / 2
        m = (c - b) / 2
        (abs(m) <= tol || iszero(f_b)) && return b
        if abs(e) >= tol && abs(f_a) > abs(f_b)
            s = f_b / f_a
            if a == c # secant
                p, q = 2 * m * s, 1 - s
            else # inverse quadratic interpolation
                q, r = f_a / f_c, f_b / f_c
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            end
            p > 0 ? (q = -q) : (p = -p)
            if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q))
                e, d = d, p / q
            else
                d = e = m
            end
        else
            d = e = m
        end
        a, f_a = b, f_b
        b += abs(d) > tol ? d : copysign(tol, m)
        f_b = f(b)
    end
    return b
end

# Supremum of the s > 0 such that E[exp(mul * s * X)] is finite, i.e. characteristic_exponent_i(mul * s, model) is defined:
# beyond it the exponent returns meaningless finite values and the Chernoff bound does not hold.
exponential_moment_bound(model, mul) = Inf
exponential_moment_bound(model::FinancialMonteCarlo.KouProcess, mul) = v_value_mod(mul > 0 ? model.λ₊ : model.λ₋)
# Positive root in s of 1 - a s² - mul b s = 0, where the argument of the log (VG) or of the sqrt (NIG) vanishes.
function quadratic_moment_bound(a, b, mul)
    a, b = v_value_mod(a), v_value_mod(b)
    return (sqrt(b^2 + 4 * a) - mul * b) / (2 * a)
end
exponential_moment_bound(model::FinancialMonteCarlo.VarianceGammaProcess, mul) = quadratic_moment_bound(model.σ^2 * model.κ / 2, model.θ * model.κ, mul)
exponential_moment_bound(model::FinancialMonteCarlo.NormalInverseGaussianProcess, mul) = quadratic_moment_bound(model.σ^2 * model.κ, 2 * model.θ * model.κ, mul)

# The bound of a Lévy process does not depend on the maturity.
exponential_moment_bound(model, mul, T) = exponential_moment_bound(model, mul)
# Inverse of the time at which E[exp(z X_t)] explodes for a real z, i.e. of the first zero of
# cosh(γ t / 2) + β sinh(γ t / 2) / γ: zero where the moments never explode. Continuous and finite everywhere, also at
# D = 0, where both branches tend to -β / 2 (to 0 for β > 0): no Inf to divide by in the residual of the bound.
function heston_explosion_rate(z, κ_s, ρσ, σ²)
    0 <= z <= 1 && return zero(κ_s)
    β = κ_s - ρσ * z
    D = κ_s^2 + z * (σ² - 2 * κ_s * ρσ + z * (ρσ^2 - σ²))
    if D < 0
        q = sqrt(-D)
        return q / (2 * atan(q, -β))
    end
    β >= 0 && return zero(κ_s)
    γ = sqrt(D)
    return γ / log((β - γ) / (β + γ))
end
# The moments of Heston explode in finite time: the domain of E[exp(mul * s * X_T)] is an interval, shrinking as T grows.
# Its end solves T / τ(s) = 1 for the explosion time τ, written with the rate 1 / τ (zero before the s where τ becomes
# finite): T / τ - 1 is continuous, -1 before that s and increasing after, so the doubling
# from s = 1 brackets the root and Brent's method solves it: 0.7 µs against 4.3 µs for a bisection to the same tolerance,
# which was half the Heston range after issue #38. Returns an s where the moments are still finite.
function exponential_moment_bound(model::FinancialMonteCarlo.HestonProcess, mul, T, max_doublings = 60, toll = 1e-12)
    κ_s, ρσ, σ² = v_value_mod(model.κ + model.λ), v_value_mod(model.ρ * model.σ), v_value_mod(model.σ^2)
    residual(s) = T * heston_explosion_rate(mul * s, κ_s, ρσ, σ²) - 1
    s_low, s_high = zero(κ_s), one(κ_s)
    f_low, f_high = -one(κ_s), residual(s_high)
    n = 0
    while signbit(f_high)
        n == max_doublings && return oftype(s_high, Inf)
        s_low, s_high, f_low = s_high, 2 * s_high, f_high
        f_high = residual(s_high)
        n += 1
    end
    s = brent_root(residual, s_low, s_high, f_low, f_high, toll * s_high)
    # Brent's estimate is within its tolerance of the root, on either side: step back to the finite side
    while !signbit(residual(s))
        s -= 2 * eps(s) + toll * s
    end
    return s
end

# The s minimising t(s), a constant for differentiation (exponent and log_epss without derivatives), to a relative
# tolerance toll. Any admissible s gives a valid bound, t(s) >= t(s*), and t'(s*) = 0, so an error δ in s moves t(s) by
# O(t''(s*) δ²) only: a tolerance sqrt(eps) gives t to about eps (issue #38).
# The bracket starts at s_min (if the bound is already increasing there, the minimum is below it, for very long maturities
# or large volatilities, and t(s_min) is still a valid, wider, bound) and at the Gaussian guess from s_min, beyond the root
# for the heavier tails of every model in the package; while it is not, the upper end moves to the next guess, at least
# doubled, never beyond the exponential moment bound B (if t still decreases there, the bound itself is the best
# admissible s). A guess where the exponent overflows (g not finite: far beyond the root, where κ is of the order of L)
# is halved towards the lower end.
# Then Brent's method, in the coordinates u = log(s / (B - s)) (log s for B = ∞) and log(g - g0) (g0 = 0 for the tails,
# g0 < g(s_lo) for an exponent with g(s_lo) <= 0, the aliasing one of fft_parameters.jl), which keep the root and the
# signs: there g is a line both for the Gaussian (slope 2) and near a pole of the exponent at B (Kou, VG, NIG, Heston).
# On g itself the interpolation lands next to the lower end, step after step, and ends in bisection (g grows like
# exp(s²) in Merton, and the root of NIG is 4e-6 below B relative): 12-30 evaluations of the exponent per range (both
# tails) on the models of the tests against 102-115 of a bisection to 1e-14 in [1, 50]. A tolerance toll on u is one
# on log s, since d log s = du (B - s) / B.
function chernoff_optimum(exponent, log_epss, s_min, toll, max_steps = 60)
    FT = typeof(s_min)
    L = -log_epss
    B = convert(FT, exponential_moment_bound(exponent))
    # never evaluate at B, where the exponent is infinite or meaningless; sqrt(eps) keeps it below B in Float32 too
    s_bound = B * (1 - sqrt(eps(FT)))
    g(s) = chernoff_g(s, exponent)
    s_lo = min(s_min, s_bound / 2)
    g_lo = g(s_lo)
    g_lo < L || return s_lo
    s_hi = min(gaussian_root_guess(s_lo, g_lo, L), s_bound)
    g_hi = g(s_hi)
    for _ = 1:max_steps
        if !isfinite(g_hi)
            s_hi = (s_lo + s_hi) / 2
        elseif g_hi < L
            s_hi == s_bound && return s_hi
            s_lo, g_lo = s_hi, g_hi
            s_hi = min(max(2 * s_hi, gaussian_root_guess(s_hi, g_hi, L)), s_bound)
        else
            break
        end
        g_hi = g(s_hi)
    end
    (isfinite(g_hi) && g_hi >= L) || return s_lo
    g0 = min(zero(g_lo), 2 * g_lo)
    # g increases on the bracket, so g - g0 >= g_lo - g0 > 0 there, up to rounding when g_lo is about 0. One log of the
    # ratio, the gap computed once: near the root the ratio is about 1, and its log is accurate to about eps, where
    # log(g - g0) - log(L - g0) would lose eps |log(L - g0)| to the cancellation.
    gap = L - g0
    log_gap(g_s) = log(max(g_s - g0, eps(L)) / gap)
    residual(s) = log_gap(g(s))
    # the coordinate is chosen once, by dispatch; both branches return FT, so the result stays type stable
    r_lo, r_hi = log_gap(g_lo), log_gap(g_hi)
    isfinite(B) && return brent_in_coordinate(BoundedCoordinate(B), residual, s_lo, s_hi, r_lo, r_hi, toll)
    return brent_in_coordinate(UnboundedCoordinate(), residual, s_lo, s_hi, r_lo, r_hi, toll)
end

# The coordinate u of Brent's method: u = log(s / (B - s)) below a finite bound B, u = log(s) without one.
struct BoundedCoordinate{T}
    B::T
end
struct UnboundedCoordinate end
to_u(c::BoundedCoordinate, s) = log(s / (c.B - s))
to_s(c::BoundedCoordinate, u) = c.B / (1 + exp(-u))
to_u(::UnboundedCoordinate, s) = log(s)
to_s(::UnboundedCoordinate, u) = exp(u)
# The root in s of residual(s), found in the coordinate u: a function of its own, compiled for each coordinate.
brent_in_coordinate(c, residual, s_lo, s_hi, r_lo, r_hi, toll) = to_s(c, brent_root(u -> residual(to_s(c, u)), to_u(c, s_lo), to_u(c, s_hi), r_lo, r_hi, toll))

function compute_chernoff_extrema_single(Model, T, log_epss, s_min, toll, mul, driftT)
    s_opt = ChainRulesCore.@ignore_derivatives chernoff_optimum(TailExponent(Model, v_value_mod(T), mul), v_value_mod(log_epss), s_min, toll)
    # t at the s found, not κ'(s), which equals t(s*) at the optimum only and is not a bound elsewhere
    t_opt = compute_ts(s_opt, Model, T, log_epss, mul)
    # mul = 1 is the upper tail: the bound on E[exp(s X)] limits how far X goes above driftT
    return driftT + mul * t_opt
end

function compute_chernoff_extrema(Model, T, epss, drift, s_min = 1.0, toll = sqrt(eps(Float64)))
    log_epss = log(epss)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model, T)
    mul_plus = 1
    t_opt_plus = compute_chernoff_extrema_single(Model, T, log_epss, s_min, toll, mul_plus, driftT)
    mul_minus = -1
    t_opt_neg = compute_chernoff_extrema_single(Model, T, log_epss, s_min, toll, mul_minus, driftT)
    return t_opt_neg, t_opt_plus
end

# Truncation range [a, b] of log(S_T / S0), for the COS methods and the FFT density: a tail probability of 1e-14 in
# Float64 on each side.
function compute_chernorff_limits(Model::FinancialMonteCarlo.BaseProcess, rT, dT, T)
    drift = rT - dT
    FT = float_type(drift, T, Model.underlying.S0)
    # the tail is weighted by the scores in the derivatives: the error of the second one in σ is about 2500 ε (1e-14
    # in Float64, scaled by the machine epsilon of the other types)
    epss = scaled_tolerance(FT, 1e-14)
    s_min = one(FT)
    compute_chernoff_extrema(Model, T, epss, drift, s_min, sqrt(eps(FT)))
end
