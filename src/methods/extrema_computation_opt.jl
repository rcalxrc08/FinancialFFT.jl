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

function compute_ts_derivative_full_complex(s, exponent, log_epss)
    h = complex_step(s, exponent)
    s_complex = @muladd s + im * h
    t_s_complex = compute_ts(s_complex, exponent, log_epss)
    t_s_epsilon = v_value_mod(FinancialFFT.imag_mod(t_s_complex)) / h
    return t_s_epsilon
end

# t(s) = (T ψ(s) - log ε) / s has t'(s) = (T (s ψ'(s) - ψ(s)) + log ε) / s², whose numerator is increasing for a convex ψ:
# t' changes sign at most once, from negative to positive, so the only stationary point is the minimum and
# the bracket [a, b] keeps t'(a) < 0 ≤ t'(b).
function optimize_ts_with_bisection(exponent, log_epss, toll_, a, b, f_a = compute_ts_derivative_full_complex(a, exponent, log_epss), f_b = compute_ts_derivative_full_complex(b, exponent, log_epss))
    @assert (b > a && signbit(f_a) && !signbit(f_b)) "not possible"
    x_half = (a + b) / 2
    # relative tolerance: for short maturities the optimum is large and the absolute one is below the float spacing
    if b - a < toll_ * max(one(b), b)
        return x_half
    end
    f_x_half = compute_ts_derivative_full_complex(x_half, exponent, log_epss)
    if signbit(f_x_half)
        return optimize_ts_with_bisection(exponent, log_epss, toll_, x_half, b, f_x_half, f_b)
    end
    return optimize_ts_with_bisection(exponent, log_epss, toll_, a, x_half, f_a, f_x_half)
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
# Time at which E[exp(z X_t)] explodes for a real z, i.e. the first zero of cosh(γ t / 2) + β sinh(γ t / 2) / γ.
function heston_explosion_time(z, κ_s, ρσ, σ²)
    0 <= z <= 1 && return Inf
    β = κ_s - ρσ * z
    D = κ_s^2 + z * (σ² - 2 * κ_s * ρσ + z * (ρσ^2 - σ²))
    if D < 0
        q = sqrt(-D)
        return 2 * atan(q, -β) / q
    end
    β >= 0 && return Inf
    γ = sqrt(D)
    return log((β - γ) / (β + γ)) / γ
end
# The moments of Heston explode in finite time: the domain of E[exp(mul * s * X_T)] is an interval, shrinking as T grows.
function exponential_moment_bound(model::FinancialMonteCarlo.HestonProcess, mul, T, max_doublings = 60, toll = 1e-12)
    κ_s, ρσ, σ² = v_value_mod(model.κ + model.λ), v_value_mod(model.ρ * model.σ), v_value_mod(model.σ^2)
    is_finite(s) = heston_explosion_time(mul * s, κ_s, ρσ, σ²) > T
    s_low, s_high = 0.0, 1.0
    n = 0
    while is_finite(s_high)
        n == max_doublings && return Inf
        s_low, s_high = s_high, 2 * s_high
        n += 1
    end
    while s_high - s_low > toll * s_high
        s_half = (s_low + s_high) / 2
        s_low, s_high = is_finite(s_half) ? (s_half, s_high) : (s_low, s_half)
    end
    return s_low
end

# The optimal s grows as the maturity shrinks (s* = sqrt(-2 log(eps) / (σ² T)) in Black Scholes),
# so the upper end of the bracket is doubled until it contains the minimum, without crossing the exponential moment bound.
# Returns the bracket and whether it contains the minimum; when it does not, the upper end is the best admissible s.
function widen_bisection_bracket(exponent, log_epss, s_min, s_max, max_doublings = 30)
    s_bound = convert(typeof(s_max), (1 - 1e-8) * exponential_moment_bound(exponent))
    s_max = min(s_max, s_bound)
    s_min = min(s_min, s_max / 2)
    is_increasing(s) = !signbit(compute_ts_derivative_full_complex(s, exponent, log_epss))
    # the minimum is below s_min (very long maturities or large volatilities): t(s_min) is still a valid, wider, bound
    is_increasing(s_min) && return s_min, s_min, false
    n = 0
    while !is_increasing(s_max) && s_max < s_bound && n < max_doublings
        s_max = min(2 * s_max, s_bound)
        n += 1
    end
    return s_min, s_max, is_increasing(s_max)
end

# The s minimising t(s), a constant for differentiation (exponent and log_epss without derivatives).
function chernoff_optimum(exponent, log_epss, s_min, s_max, x_toll_bisection)
    s_min_adj, s_max_adj, is_bracketed = widen_bisection_bracket(exponent, log_epss, s_min, s_max)
    # if the bound is still decreasing at the exponential moment bound, the best admissible s is the bound itself;
    # if it is already increasing at s_min, it is s_min
    return is_bracketed ? optimize_ts_with_bisection(exponent, log_epss, x_toll_bisection, s_min_adj, s_max_adj) : s_max_adj
end

function compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul, driftT)
    s_opt = ChainRulesCore.@ignore_derivatives chernoff_optimum(TailExponent(Model, v_value_mod(T), mul), v_value_mod(log_epss), s_min, s_max, x_toll_bisection)
    t_opt_bis = compute_ts(s_opt, Model, T, log_epss, mul)
    # mul = 1 is the upper tail: the bound on E[exp(s X)] limits how far X goes above driftT
    return driftT + mul * t_opt_bis
end

function compute_chernoff_extrema_bisection(Model, T, epss, drift, s_min = 1.0, s_max = 50.0, x_toll_bisection = 1e-14)
    log_epss = log(epss)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model, T)
    mul_plus = 1
    t_opt_bis_plus = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul_plus, driftT)
    mul_minus = -1
    t_opt_bis_neg = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul_minus, driftT)
    return t_opt_bis_neg, t_opt_bis_plus
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
    s_max = FT(50)
    x_toll_bisection = max(FT(1e-14), 8 * eps(FT))
    compute_chernoff_extrema_bisection(Model, T, epss, drift, s_min, s_max, x_toll_bisection)
end
