using FinancialFFT, FinancialMonteCarlo, DualNumbers, HyperDualNumbers, MuladdMacro, ChainRulesCore

function compute_ts(s, model, T, log_epss, mul)
    return @muladd inv(s) * (T * FinancialFFT.characteristic_exponent_i(mul * s, model) - log_epss)
end

function compute_ts_differentiable(s, model, T, log_epss, mul)
    return @muladd s * inv(log_epss - T * FinancialFFT.characteristic_exponent_i(mul * s, model))
end

function compute_ts2(s, model, T, log_epss, mul)
    s_adj = abs(s)
    return @muladd inv(s_adj) * (T * FinancialFFT.characteristic_exponent_i(mul * s_adj, model) - log_epss)
end

function compute_ts_and_derivative(s, model, T, log_epss, mul)
    s_dual = dual(s, one(s))
    t_s = compute_ts(s_dual, model, T, log_epss, mul)
    return t_s.value, t_s.epsilon
end
function compute_ts_differentiable_and_derivative(s, model, T, log_epss, mul)
    s_dual = dual(s, one(s))
    t_s = compute_ts_differentiable(s_dual, model, T, log_epss, mul)
    return -inv(t_s.value), t_s.epsilon
end

function compute_ts_and_derivative_complex(s, model, T, log_epss, mul)
    h = eps(s)
    s_complex = complex(s, h)
    t_s_complex = compute_ts(s_complex, model, T, log_epss, mul)
    t_s_value = real(t_s_complex)
    t_s_epsilon = imag(t_s_complex) / h
    return t_s_value, t_s_epsilon
end

function compute_ts_and_newton_iteration(s, model, T, log_epss, mul)
    t_s = compute_ts(s, model, T, log_epss, mul)
    return t_s.value, -t_s.epsilon1 / t_s.epsilon12
end

function compute_ts_diff_and_newton_iteration(s, model, T, log_epss, mul)
    t_s = compute_ts_differentiable(s, model, T, log_epss, mul)
    return -inv(t_s.value), -t_s.epsilon1 / t_s.epsilon12
end

function optimize_ts_with_newton_method(s_0, model, T, log_epss, xtoll, nmax, mul)
    one_s = one(s_0)
    zero_s = zero(s_0)
    s_hyper_i = hyper(s_0, one_s, one_s, zero_s)
    one_adj = hyper(one_s, one_s, one_s, zero_s)
    for _ = 1:nmax
        t_i, incr = compute_ts_and_newton_iteration(s_hyper_i, model, T, log_epss, mul)
        diff = abs(incr)
        if (diff < xtoll)
            return s_hyper_i.value, t_i
        end
        s_hyper_i += incr
        s_hyper_i = max(s_hyper_i, one_adj)
    end
    s_min = one_s
    s_max = 50 * one_s
    # Reverting to bisection method.
    return optimize_ts_with_bisection(model, T, log_epss, xtoll, mul, s_min, s_max)
end

function optimize_s_with_newton_method(s_0, model, T, log_epss, xtoll, nmax, mul)
    one_s = one(s_0)
    zero_s = zero(s_0)
    s_hyper_i = hyper(s_0, one_s, one_s, zero_s)
    one_adj = hyper(one_s, one_s, one_s, zero_s)
    for _ = 1:nmax
        t_i, incr = compute_ts_and_newton_iteration(s_hyper_i, model, T, log_epss, mul)
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
    s_opt, _ = optimize_ts_with_bisection(model, T, log_epss, xtoll, mul, s_min, s_max)
    return s_opt
end

function optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, b, ffa = compute_ts_and_derivative(a, Model, T, log_epss, mul), ffb = compute_ts_and_derivative(b, Model, T, log_epss, mul), n_iter = 0, a_0 = a, b_0 = b)
    _, f_a = ffa
    _, f_b = ffb
    b_a = b - a
    sign_f_a = signbit(f_a)
    sign_f_b = signbit(f_b)
    @assert (b_a > 0 && sign_f_a != sign_f_b) "not possible"
    x_half = (a + b) / 2
    arg_x_half = compute_ts_and_derivative(x_half, Model, T, log_epss, mul)
    t_x_half, f_x_half = arg_x_half
    if b_a < toll_
        #I should check if it is a maxima or a minima. In case of maxima, I should run again the bisection in one of the two sections [a,x*-eps] or [x*+eps,b]
        is_minima = sign_f_a && !sign_f_b
        if (is_minima)
            return x_half, t_x_half
        end
        ffa_0 = compute_ts_and_derivative(a_0, Model, T, log_epss, mul)
        _, f_a0 = ffa_0
        if (sign_f_a && !signbit(f_a0))
            return optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, a_0, ffa, ffa_0, n_iter + 1, a_0, b_0)
        end
        ffb_0 = compute_ts_and_derivative(b_0, Model, T, log_epss, mul)
        return optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, b_0, b, ffb_0, ffb, n_iter + 1, a_0, b_0)
    end
    sign_f_x_half = signbit(f_x_half)
    if sign_f_a == sign_f_x_half
        optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, x_half, b, arg_x_half, ffb, n_iter + 1, a_0, b_0)
    else
        optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, x_half, ffa, arg_x_half, n_iter + 1, a_0, b_0)
    end
end

function optimize_ts_with_bisection_just_result(Model, T, log_epss, toll_, mul, a, b, ffa = compute_ts_and_derivative(a, Model, T, log_epss, mul), ffb = compute_ts_and_derivative(b, Model, T, log_epss, mul), n_iter = 0, a_0 = a, b_0 = b)
    _, f_a = optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, b, ffa, ffb, n_iter, a_0, b_0)
    return f_a
end

function optimize_ts_with_bisection_just_s(Model, T, log_epss, toll_, mul, a, b, ffa = compute_ts_and_derivative(a, Model, T, log_epss, mul), ffb = compute_ts_and_derivative(b, Model, T, log_epss, mul), n_iter = 0, a_0 = a, b_0 = b)
    s_opt, _ = optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, b, ffa, ffb, n_iter, a_0, b_0)
    return s_opt
end
using ChainRulesCore: rrule_via_ad
function common_pullback(slice, config, s_opt, Model, T, log_epss, mul)
    _, pullback_ts = ChainRulesCore.rrule_via_ad(config, compute_ts, s_opt, Model, T, log_epss, mul)
    _, _, der_Model, der_T, der_log_epss, _ = pullback_ts(slice)
    return der_Model, der_T, der_log_epss
end

function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(optimize_ts_with_bisection_just_result), Model, T, log_epss, toll_, mul, a, b, ffa, ffb, n_iter, a_0, b_0)
    s_opt, t_opt = optimize_ts_with_bisection(Model, T, log_epss, toll_, mul, a, b, ffa, ffb, n_iter, a_0, b_0)
    function update_pullback(slice)
        # _, pullback_ts = ChainRulesCore.rrule_via_ad(config, compute_ts, s_opt, Model, T, log_epss, mul)
        # _, _, der_Model, der_T, der_log_epss, _ = pullback_ts(slice)
        der_Model, der_T, der_log_epss = common_pullback(slice, config, s_opt, Model, T, log_epss, mul)
        return NoTangent(), der_Model, der_T, der_log_epss, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return t_opt, update_pullback
end

function optimize_ts_with_newton_method_just_result(s_0, model, T, log_epss, xtoll, nmax, mul)
    _, t_opt = optimize_ts_with_newton_method(s_0, model, T, log_epss, xtoll, nmax, mul)
    return t_opt
end

function optimize_ts_with_newton_method_just_s(s_0, model, T, log_epss, xtoll, nmax, mul)
    s, _ = optimize_ts_with_newton_method(s_0, model, T, log_epss, xtoll, nmax, mul)
    return s
end

function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(optimize_ts_with_newton_method_just_result), s_0, Model, T, log_epss, xtoll, nmax, mul)
    s_opt, t_opt = optimize_ts_with_newton_method(s_0, Model, T, log_epss, xtoll, nmax, mul)
    function update_pullback(slice)
        # _, pullback_ts = ChainRulesCore.rrule_via_ad(config, compute_ts, s_opt, Model, T, log_epss, mul)
        # _, _, der_Model, der_T, der_log_epss, _ = pullback_ts(slice)
        der_Model, der_T, der_log_epss = common_pullback(slice, config, s_opt, Model, T, log_epss, mul)
        return NoTangent(), NoTangent(), der_Model, der_T, der_log_epss, NoTangent(), NoTangent(), NoTangent()
    end
    return t_opt, update_pullback
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
function compute_positive_extrema(Model, T, epss, drift)
    log_epss = log(epss)
    s_min = 1.0
    s_max = 500.0
    x_toll_bisection = 1e-14
    t_plus = optimize_ts_with_bisection_just_result(Model, T, log_epss, x_toll_bisection, -1, s_min, s_max)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return t_plus + driftT
end
function compute_positive_extrema_newton(Model, T, epss, drift)
    log_epss = log(epss)
    x_toll_newton = 1e-14
    s_0 = 10.0
    nmax = 3000
    t_opt_newton = optimize_ts_with_newton_method_just_result(s_0, Model, T, log_epss, x_toll_newton, nmax, -1)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return t_opt_newton + driftT
end
function compute_negative_extrema(Model, T, epss, drift)
    log_epss = log(epss)
    s_min = 20.0
    s_max = 450.0
    x_toll_bisection = 1e-14
    t_plus = optimize_ts_with_bisection_just_result(Model, T, log_epss, x_toll_bisection, 1, s_min, s_max)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_plus + driftT
end
function compute_negative_extrema_newton(Model, T, epss, drift)
    log_epss = log(epss)
    x_toll_newton = 1e-14
    s_0 = 20.0
    nmax = 300
    t_opt_newton = optimize_ts_with_newton_method_just_result(s_0, Model, T, log_epss, x_toll_newton, nmax, 1)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_opt_newton + driftT
end
function compute_extrema_bisection(Model, T, epss, drift)
    log_epss = log(epss)
    s_min = 1.0
    s_max = 50.0
    x_toll_bisection = 1e-14
    t_plus_plus = optimize_ts_with_bisection_just_result(Model, T, log_epss, x_toll_bisection, -1, s_min, s_max)
    t_plus_neg = optimize_ts_with_bisection_just_result(Model, T, log_epss, x_toll_bisection, 1, s_min, s_max)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_plus_neg + driftT, t_plus_plus + driftT
end
function compute_extrema_newton_full(Model, T, epss, drift, s_0, x_toll_newton, nmax)
    log_epss = log(epss)
    t_opt_newton_plus = optimize_ts_with_newton_method_just_result(s_0, Model, T, log_epss, x_toll_newton, nmax, -1)
    t_opt_newton_neg = optimize_ts_with_newton_method_just_result(s_0, Model, T, log_epss, x_toll_newton, nmax, 1)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_opt_newton_neg + driftT, t_opt_newton_plus + driftT
end
function compute_extrema_newton(Model, T, epss, drift)
    x_toll_newton = 1e-14
    s_0 = 4.0
    nmax = 300
    return compute_extrema_newton_full(Model, T, epss, drift, s_0, x_toll_newton, nmax)
end
K = 100.0;
rT_dT = FinancialMonteCarlo.integral(r - Model.underlying.d, T)
@show b = compute_positive_extrema(Model, T, epss, rT_dT + log(S0 / K))
@show b = compute_positive_extrema_newton(Model, T, epss, rT_dT + log(S0 / K))
@show a = compute_negative_extrema(Model, T, epss, rT_dT + log(S0 / K))
@show a = compute_negative_extrema_newton(Model, T, epss, rT_dT + log(S0 / K))
@show x = compute_extrema_bisection(Model, T, epss, rT_dT + log(S0 / K))
@show x = compute_extrema_newton(Model, T, epss, rT_dT + log(S0 / K))
using BenchmarkTools
# @btime compute_positive_extrema($Model, $T, $epss, $rT_dT)
# @btime compute_positive_extrema_newton($Model, $T, $epss, $rT_dT)
# @btime compute_negative_extrema($Model, $T, $epss, $rT_dT)
# @btime compute_negative_extrema_newton($Model, $T, $epss, $rT_dT)
# @btime compute_extrema_bisection($Model, $T, $epss, $rT_dT)
# @btime compute_extrema_newton($Model, $T, $epss, $rT_dT)
function compute_extrema_newton_full_from_param(sigma, lam, mu1, sigma1, S0, d, K, T, epss, s_0, x_toll_newton, nmax)
    Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d))
    rT_dT = FinancialMonteCarlo.integral(r - d, T)
    drift = rT_dT + log(S0 / K)
    log_epss = log(epss)
    t_opt_newton_plus = ChainRulesCore.@ignore_derivatives optimize_ts_with_newton_method_just_result(s_0, Model, T, log_epss, x_toll_newton, nmax, -1)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return t_opt_newton_plus + driftT
end
function compute_extrema_newton_full_from_param22(sigma, lam, mu1, sigma1, S0, r, d, K, T, epss, s_0, x_toll_newton, nmax)
    Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d))
    rT_dT = FinancialMonteCarlo.integral(r - d, T)
    drift = rT_dT + log(S0 / K)
    log_epss = log(epss)
    mul = -1
    # @show Model
    s_opt = ChainRulesCore.@ignore_derivatives optimize_ts_with_newton_method_just_s(s_0, Model, T, log_epss, x_toll_newton, nmax, mul)
    # @show s_opt
    t_opt_newton_plus = compute_ts(s_opt, Model, T, log_epss, mul)
    driftT = @muladd drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return t_opt_newton_plus + driftT
end

function compute_extrema_newton_full_from_param2(sigma, lam, mu1, sigma1, S0, r, d, K, T, epss)
    x_toll_newton = 1e-14
    s_0 = 4.0
    nmax = 300
    return compute_extrema_newton_full_from_param22(sigma, lam, mu1, sigma1, S0, r, d, K, T, epss, s_0, x_toll_newton, nmax)
end
@btime compute_extrema_newton_full_from_param2($sigma, $lam, $mu1, $sigma1, $S0, $r, $d, $K, $T, $epss);
using Zygote, FiniteDiff
@show Zygote.gradient(compute_extrema_newton_full_from_param2, sigma, lam, mu1, sigma1, S0, r, d, K, T, epss)
@show FiniteDiff.finite_difference_gradient(x -> compute_extrema_newton_full_from_param2(x..., epss), [sigma, lam, mu1, sigma1, S0, d, K, T])
@btime Zygote.gradient(compute_extrema_newton_full_from_param2, $sigma, $lam, $mu1, $sigma1, $S0, $r, $d, $K, $T, $epss)