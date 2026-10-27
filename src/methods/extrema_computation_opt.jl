using FinancialMonteCarlo, MuladdMacro, ChainRulesCore

function compute_ts(s, model, T, log_epss, mul)
    return @muladd inv(s) * (T * FinancialFFT.characteristic_exponent_i(mul * s, model) - log_epss)
end

function v_value_mod(s::AbstractFloat)
    return s
end

function compute_ts_derivative_full_complex(s, model, T, log_epss, mul)
    h = eps(s)
    s_complex = @muladd s + im * h
    t_s_complex = compute_ts(s_complex, model, T, log_epss, mul)
    t_s_epsilon = v_value_mod(FinancialFFT.imag_mod(t_s_complex)) / h
    return t_s_epsilon
end

function optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, b, f_a = compute_ts_derivative_full_complex(a, Model, T, log_epss, mul), f_b = compute_ts_derivative_full_complex(b, Model, T, log_epss, mul), n_iter = 0, a_0 = a, b_0 = b)
    b_a = b - a
    sign_f_a = signbit(f_a)
    sign_f_b = signbit(f_b)
    @assert (b_a > 0 && sign_f_a != sign_f_b) "not possible"
    x_half = (a + b) / 2
    f_x_half = compute_ts_derivative_full_complex(x_half, Model, T, log_epss, mul)
    # relative tolerance: for short maturities the optimum is large and the absolute one is below the float spacing
    if b_a < toll_ * max(one(b), b)
        #I should check if it is a maxima or a minima. In case of maxima, I should run again the bisection in one of the two sections [a,x*-eps] or [x*+eps,b]
        is_minima = sign_f_a && !sign_f_b
        if (is_minima)
            return x_half
        end
        f_a0 = compute_ts_derivative_full_complex(a_0, Model, T, log_epss, mul)
        if (sign_f_a && !signbit(f_a0))
            return optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, a_0, f_a, f_a0, n_iter + 1, a_0, b_0)
        end
        fb_0 = compute_ts_derivative_full_complex(b_0, Model, T, log_epss, mul)
        return optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, b_0, b, fb_0, f_b, n_iter + 1, a_0, b_0)
    end
    sign_f_x_half = signbit(f_x_half)
    if sign_f_a == sign_f_x_half
        optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, x_half, b, f_x_half, f_b, n_iter + 1, a_0, b_0)
    else
        optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, x_half, f_a, f_x_half, n_iter + 1, a_0, b_0)
    end
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

# The optimal s grows as the maturity shrinks (s* = sqrt(-2 log(eps) / (σ² T)) in Black Scholes),
# so the upper end of the bracket is doubled until it contains the minimum, without crossing the exponential moment bound.
# Returns the bracket and whether it contains a sign change of the derivative.
function widen_bisection_bracket(Model, T, log_epss, mul, s_min, s_max, max_doublings = 30)
    s_bound = (1 - 1e-8) * exponential_moment_bound(Model, mul)
    s_max = min(s_max, s_bound)
    s_min = min(s_min, s_max / 2)
    sign_f_min = signbit(compute_ts_derivative_full_complex(s_min, Model, T, log_epss, mul))
    has_sign_change(s) = signbit(compute_ts_derivative_full_complex(s, Model, T, log_epss, mul)) != sign_f_min
    n = 0
    while !has_sign_change(s_max) && s_max < s_bound && n < max_doublings
        s_max = min(2 * s_max, s_bound)
        n += 1
    end
    return s_min, s_max, has_sign_change(s_max)
end

function compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul, driftT)
    s_opt = ChainRulesCore.@ignore_derivatives begin
        T_val = v_value_mod(T)
        log_epss_val = v_value_mod(log_epss)
        s_min_adj, s_max_adj, is_bracketed = widen_bisection_bracket(Model, T_val, log_epss_val, mul, s_min, s_max)
        # if the bound is still decreasing at the exponential moment bound, the best admissible s is the bound itself
        is_bracketed ? optimize_ts_with_bisection(Model, T_val, log_epss_val, x_toll_bisection, mul, s_min_adj, s_max_adj) : s_max_adj
    end
    t_opt_bis = compute_ts(s_opt, Model, T, log_epss, mul)
    return driftT - mul * t_opt_bis
end

function compute_chernoff_extrema_bisection(Model, T, epss, drift, s_min, s_max, x_toll_bisection)
    log_epss = log(epss)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    mul_plus = -1
    t_opt_bis_plus = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul_plus, driftT)
    mul_minus = 1
    t_opt_bis_neg = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul_minus, driftT)
    return t_opt_bis_neg, t_opt_bis_plus
end

function compute_positive_extrema_bisection(Model, T, epss, drift, s_min = 1.0, s_max = 50.0, x_toll_bisection = 1e-14)
    log_epss = log(epss)
    mul = -1
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    t_plus = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul, driftT)
    return t_plus
end
function compute_negative_extrema_bisection(Model, T, epss, drift, s_min = 1.0, s_max = 50.0, x_toll_bisection = 1e-14)
    log_epss = log(epss)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    mul = 1
    t_minus = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul, driftT)
    return t_minus
end

function compute_extrema_bisection_with_default(Model, T, epss, drift, s_min = 1.0, s_max = 50.0, x_toll_bisection = 1e-14)
    res = compute_chernoff_extrema_bisection(Model, T, epss, drift, s_min, s_max, x_toll_bisection)
    return res
end