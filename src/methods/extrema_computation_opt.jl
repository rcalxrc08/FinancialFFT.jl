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
    if b_a < toll_
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

function compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul, driftT)
    s_opt = ChainRulesCore.@ignore_derivatives optimize_ts_with_bisection(Model, v_value_mod(T), v_value_mod(log_epss), x_toll_bisection, mul, s_min, s_max)
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