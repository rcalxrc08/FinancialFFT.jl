using FinancialFFT, FinancialMonteCarlo, DualNumbers, HyperDualNumbers, MuladdMacro, ChainRulesCore

function compute_ts(s, model, T, log_epss, mul)
    return @muladd inv(s) * (T * FinancialFFT.characteristic_exponent_i(mul * s, model) - log_epss)
end

function compute_ts_derivative_analytic(s, model, T, log_epss, mul)
    inv_s = inv(s)
    t_s = compute_ts(s, model, T, log_epss, mul)
    return @muladd -inv_s * (t_s - T * characteristic_exponent_i_der(mul * s, model))
end

function compute_ts_derivative_analytic2(s, model, T, log_epss, mul)
    inv_s = inv(s)
    der_t_s = compute_ts_derivative_analytic(s, model, T, log_epss, mul)
    return @muladd inv(s) * (T * characteristic_exponent_i_der_der(mul * s, model) - 2 * der_t_s)
end

function compute_ts_derivative_analytic1_2(s, model, T, log_epss, mul)
    inv_s = inv(s)
    der_t_s = compute_ts_derivative_analytic(s, model, T, log_epss, mul)
    return @muladd der_t_s * inv_s / (T * characteristic_exponent_i_der_der(mul * s, model) - 2 * der_t_s)
end

function v_value_mod(s::AbstractFloat)
    return s
end
function v_value_mod(s::Dual)
    return s.value
end
function v_value_mod(s::Hyper)
    return s.value
end
using ForwardDiff
function v_value_mod(s::ForwardDiff.Dual)
    return s.value
end

function compute_ts_derivative_full_complex(s, model, T, log_epss, mul)
    h = eps(s)
    s_complex = @muladd s + im * h
    t_s_complex = compute_ts(s_complex, model, T, log_epss, mul)
    t_s_epsilon = v_value_mod(FinancialFFT.imag_mod(t_s_complex)) / h
    return t_s_epsilon
end

function compute_ts_newton_iteration(s, model, T, log_epss, mul)
    t_s = compute_ts(s, model, T, log_epss, mul)
    return -t_s.epsilon1 / t_s.epsilon12
end

function optimize_ts_with_newton_method(s_0, model, T, log_epss, xtoll, nmax, mul)
    one_s = one(s_0)
    zero_s = zero(s_0)
    s_hyper_i = hyper(s_0, one_s, one_s, zero_s)
    one_adj = hyper(one_s, one_s, one_s, zero_s)
    for _ = 1:nmax
        incr = compute_ts_newton_iteration(s_hyper_i, model, T, log_epss, mul)
        diff = abs(incr)
        if (diff < xtoll)
            return s_hyper_i.value
        end
        s_hyper_i += incr
        s_hyper_i = max(s_hyper_i, one_adj)
    end
    s_min = one_s
    s_max = 50 * one_s
    # Reverting to bisection method.
    return optimize_ts_with_bisection(model, T, log_epss, xtoll, mul, s_min, s_max)
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

function compute_chernoff_extrema_newton_single(Model, T, log_epss, s_0, x_toll_newton, nmax, mul, driftT)
    ChainRulesCore.@ignore_derivatives @assert typeof(FinancialMonteCarlo.predict_output_type_zero(Model)) <: AbstractFloat "Unable to use newton method for non real numbers"
    s_opt = ChainRulesCore.@ignore_derivatives optimize_ts_with_newton_method(s_0, Model, v_value_mod(T), v_value_mod(log_epss), x_toll_newton, nmax, mul)
    t_opt_newton = compute_ts(s_opt, Model, T, log_epss, mul)
    return driftT - mul * t_opt_newton
end

function compute_chernoff_extrema_newton(Model, T, epss, drift, s_0, x_toll_newton, nmax)
    log_epss = log(epss)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    mul_plus = -1
    t_opt_newton_plus = compute_chernoff_extrema_newton_single(Model, T, log_epss, s_0, x_toll_newton, nmax, mul_plus, driftT)
    mul_neg = 1
    t_opt_newton_neg = compute_chernoff_extrema_newton_single(Model, T, log_epss, s_0, x_toll_newton, nmax, mul_neg, driftT)
    return t_opt_newton_neg, t_opt_newton_plus
end

epss = 1e-14
S0 = 100.0
r = 0.02
d = 0.01
T = 1.1
sigma = 0.2
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
# Model = BlackScholesProcess(sigma, Underlying(S0, d));
Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d))
function compute_positive_extrema_bisection(Model, T, epss, drift)
    log_epss = log(epss)
    s_min = 1.0
    s_max = 500.0
    x_toll_bisection = 1e-14
    mul = -1
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    t_plus = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul, driftT)
    return t_plus
end
function compute_positive_extrema_newton(Model, T, epss, drift)
    log_epss = log(epss)
    x_toll_newton = 1e-14
    s_0 = 10.0
    nmax = 3000
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    mul = -1
    t_opt_newton = compute_chernoff_extrema_newton_single(Model, T, log_epss, s_0, x_toll_newton, nmax, mul, driftT)
    return t_opt_newton
end
function compute_negative_extrema_bisection(Model, T, epss, drift)
    log_epss = log(epss)
    s_min = 20.0
    s_max = 450.0
    x_toll_bisection = 1e-14
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    mul = 1
    t_minus = compute_chernoff_extrema_bisection_single(Model, T, log_epss, s_min, s_max, x_toll_bisection, mul, driftT)
    return t_minus
end
function compute_negative_extrema_newton(Model, T, epss, drift)
    log_epss = log(epss)
    x_toll_newton = 1e-14
    s_0 = 20.0
    nmax = 300
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    mul = 1
    t_opt_newton = compute_chernoff_extrema_newton_single(Model, T, log_epss, s_0, x_toll_newton, nmax, mul, driftT)
    return t_opt_newton
end
function compute_extrema_bisection_with_default(Model, T, epss, drift)
    s_min = 1.0
    s_max = 50.0
    x_toll_bisection = 1e-14
    res = compute_chernoff_extrema_bisection(Model, T, epss, drift, s_min, s_max, x_toll_bisection)
    return res
end
function compute_extrema_newton_with_default(Model, T, epss, drift)
    x_toll_newton = 1e-14
    s_0 = 4.0
    nmax = 300
    return compute_chernoff_extrema_newton(Model, T, epss, drift, s_0, x_toll_newton, nmax)
end
K = 100.0;
rT_dT = FinancialMonteCarlo.integral(r - Model.underlying.d, T)
@show b = compute_positive_extrema_bisection(Model, T, epss, rT_dT + log(S0 / K))
@show b = compute_positive_extrema_newton(Model, T, epss, rT_dT + log(S0 / K))
@show a = compute_negative_extrema_bisection(Model, T, epss, rT_dT + log(S0 / K))
@show a = compute_negative_extrema_newton(Model, T, epss, rT_dT + log(S0 / K))
@show x = compute_extrema_bisection_with_default(Model, T, epss, rT_dT + log(S0 / K))
@show x = compute_extrema_newton_with_default(Model, T, epss, rT_dT + log(S0 / K))
using BenchmarkTools
# @btime compute_positive_extrema($Model, $T, $epss, $rT_dT)
# @btime compute_positive_extrema_newton($Model, $T, $epss, $rT_dT)
# @btime compute_negative_extrema($Model, $T, $epss, $rT_dT)
# @btime compute_negative_extrema_newton($Model, $T, $epss, $rT_dT)
# @btime compute_extrema_bisection($Model, $T, $epss, $rT_dT)
# @btime compute_extrema_newton($Model, $T, $epss, $rT_dT)

function compute_extrema_bisection_full_from_param(sigma, lam, mu1, sigma1, S0, r, d, K, T, epss)
    Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d))
    rT_dT = FinancialMonteCarlo.integral(r - d, T)
    drift = rT_dT + log(S0 / K)
    t_opt = compute_positive_extrema_bisection(Model, T, epss, drift)
    return t_opt
end

@btime compute_extrema_bisection_full_from_param($sigma, $lam, $mu1, $sigma1, $S0, $r, $d, $K, $T, $epss);
using Zygote, FiniteDiff
@show Zygote.gradient(compute_extrema_bisection_full_from_param, sigma, lam, mu1, sigma1, S0, r, d, K, T, epss)
@show FiniteDiff.finite_difference_gradient(x -> compute_extrema_bisection_full_from_param(x..., epss), [sigma, lam, mu1, sigma1, S0, r, d, K, T])
@btime Zygote.gradient(compute_extrema_bisection_full_from_param, $sigma, $lam, $mu1, $sigma1, $S0, $r, $d, $K, $T, $epss)
