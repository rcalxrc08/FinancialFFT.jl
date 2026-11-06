using FinancialFFT, FinancialMonteCarlo, DualNumbers

function null_der_positive(s, model, T, log_epss)
    s_dual = dual(s, one(s))
    t_s = inv(s_dual) * (T * FinancialFFT.characteristic_exponent_i(-s_dual, model) - log_epss)
    return t_s.value, t_s.epsilon
end
function null_der_negative(s, model, T, log_epss)
    s_dual = dual(s, one(s))
    t_s = inv(s_dual) * (T * FinancialFFT.characteristic_exponent_i(s_dual, model) - log_epss)
    return t_s.value, t_s.epsilon
end
using HyperDualNumbers
function null_der_positive_over_der2(s, model, T, log_epss)
    one_s = one(s)
    zero_s = zero(s)
    s_hyper = hyper(s, one_s, one_s, zero_s)
    t_s = inv(s_hyper) * (T * FinancialFFT.characteristic_exponent_i(-s_hyper, model) - log_epss)
    return t_s.value, -t_s.epsilon1 / t_s.epsilon12
end
function null_der_negative_over_der2(s, model, T, log_epss)
    one_s = one(s)
    zero_s = zero(s)
    s_hyper = hyper(s, one_s, one_s, zero_s)
    t_s = inv(s_hyper) * (T * FinancialFFT.characteristic_exponent_i(s_hyper, model) - log_epss)
    return t_s.value, -t_s.epsilon1 / t_s.epsilon12
end
function newton_new(s_0, model, T, log_epss, xtoll, nmax)
    s_i = s_0
    t_i = 0
    for _ = 1:nmax
        t_i, incr = null_der_positive_over_der2(s_i, model, T, log_epss)
        diff = abs(incr)
        if (diff < xtoll)
            return t_i
        end
        s_i += incr
    end
    @show "ERROR"
    return t_i
end

function newton_new_neg(s_0, model, T, log_epss, xtoll, nmax)
    s_i = s_0
    t_i = 0
    for _ = 1:nmax
        t_i, incr = null_der_negative_over_der2(s_i, model, T, log_epss)
        diff = abs(incr)
        s_i_1 = s_i + incr
        if (diff < xtoll)
            return t_i
        end
        s_i = s_i_1
        # if s_i < 0
        #     s_i = 10.0
        # end
    end
    @show "ERROR"
    return t_i
end

function bisection(f_::F, toll_, a, b, ffa = f_(a), ffb = f_(b), n_iter = 0) where {F}
    # @show 
    _, f_a = ffa
    _, f_b = ffb
    x_half = (a + b) * 0.5
    b_a = b - a
    arg_x_half = f_(x_half)
    t_x_half, f_x_half = arg_x_half
    if abs(b_a) < toll_
        return t_x_half
    end
    sign_f_a = sign(f_a)
    sign_f_b = sign(f_b)
    @assert !(b_a < 0 || sign_f_a * sign_f_b > 0) "not possible"
    sign_f_x_half = sign(f_x_half)
    if sign_f_a * sign_f_x_half > 0
        bisection(f_, toll_, x_half, b, arg_x_half, ffb, n_iter + 1)
    else
        bisection(f_, toll_, a, x_half, ffa, arg_x_half, n_iter + 1)
    end
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
    f_pos_local(s) = null_der_positive(s, Model, T, log_epss)
    s_min = 1.0
    s_max = 50.0
    x_toll_bisection = 1e-14
    t_plus = bisection(f_pos_local, x_toll_bisection, s_min, s_max)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return t_plus + driftT
end
function compute_positive_extrema_newton(Model, T, epss, drift)
    log_epss = log(epss)
    x_toll_bisection = 1e-14
    s_0 = 20.0
    nmax = 300
    t_opt_newton = newton_new(s_0, Model, T, log_epss, x_toll_bisection, nmax)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return t_opt_newton + driftT
end
function compute_negative_extrema(Model, T, epss, drift)
    log_epss = log(epss)
    f_neg_local(s) = null_der_negative(s, Model, T, log_epss)
    s_min = 1.0
    s_max = 50.0
    x_toll_bisection = 1e-10
    t_plus = bisection(f_neg_local, x_toll_bisection, s_min, s_max)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_plus + driftT
end
function compute_negative_extrema_newton(Model, T, epss, drift)
    log_epss = log(epss)
    x_toll_bisection = 1e-14
    s_0 = 20.0
    nmax = 300
    t_opt_newton = newton_new_neg(s_0, Model, T, log_epss, x_toll_bisection, nmax)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_opt_newton + driftT
end
function compute_extrema_bisection(Model, T, epss, drift)
    log_epss = log(epss)
    f_pos_local(s) = null_der_positive(s, Model, T, log_epss)
    f_neg_local(s) = null_der_negative(s, Model, T, log_epss)
    s_min = 1.0
    s_max = 50.0
    x_toll_bisection = 1e-14
    t_plus_plus = bisection(f_pos_local, x_toll_bisection, s_min, s_max)
    t_plus_neg = bisection(f_neg_local, x_toll_bisection, s_min, s_max)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_plus_neg + driftT, t_plus_plus + driftT
end
function compute_extrema_newton(Model, T, epss, drift)
    log_epss = log(epss)
    x_toll_newton = 1e-14
    s_0 = 20.0
    nmax = 300
    t_opt_newton_plus = newton_new(s_0, Model, T, log_epss, x_toll_newton, nmax)
    t_opt_newton_neg = newton_new_neg(s_0, Model, T, log_epss, x_toll_newton, nmax)
    driftT = drift - FinancialFFT.characteristic_exponent_i(1, Model) * T
    return -t_opt_newton_neg + driftT, t_opt_newton_plus + driftT
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
@btime compute_positive_extrema($Model, $T, $epss, $rT_dT)
@btime compute_positive_extrema_newton($Model, $T, $epss, $rT_dT)
@btime compute_negative_extrema($Model, $T, $epss, $rT_dT)
@btime compute_negative_extrema_newton($Model, $T, $epss, $rT_dT)
@btime compute_extrema_bisection($Model, $T, $epss, $rT_dT)
@btime compute_extrema_newton($Model, $T, $epss, $rT_dT)