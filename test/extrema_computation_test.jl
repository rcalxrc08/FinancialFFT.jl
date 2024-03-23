using FinancialFFT, FinancialMonteCarlo
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
Model = MertonProcess(sigma, lam, mu1, sigma1, FinancialMonteCarlo.Underlying(S0, d))
K = 100.0;
rT_dT = FinancialMonteCarlo.integral(r - Model.underlying.d, T)
@show b = FinancialFFT.compute_positive_extrema_bisection(Model, T, epss, rT_dT + log(S0 / K))
# @show b = FinancialFFT.compute_positive_extrema_newton(Model, T, epss, rT_dT + log(S0 / K))
@show a = FinancialFFT.compute_negative_extrema_bisection(Model, T, epss, rT_dT + log(S0 / K))
# @show a = FinancialFFT.compute_negative_extrema_newton(Model, T, epss, rT_dT + log(S0 / K))
@show x = FinancialFFT.compute_extrema_bisection_with_default(Model, T, epss, rT_dT + log(S0 / K))
# @show x = FinancialFFT.compute_extrema_newton_with_default(Model, T, epss, rT_dT + log(S0 / K))

# @btime compute_positive_extrema($Model, $T, $epss, $rT_dT)
# @btime compute_positive_extrema_newton($Model, $T, $epss, $rT_dT)
# @btime compute_negative_extrema($Model, $T, $epss, $rT_dT)
# @btime compute_negative_extrema_newton($Model, $T, $epss, $rT_dT)
# @btime compute_extrema_bisection($Model, $T, $epss, $rT_dT)
# @btime compute_extrema_newton($Model, $T, $epss, $rT_dT)

# function compute_extrema_bisection_full_from_param(sigma, lam, mu1, sigma1, S0, r, d, K, T, epss)
#     Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d))
#     rT_dT = FinancialMonteCarlo.integral(r - d, T)
#     drift = rT_dT + log(S0 / K)
#     t_opt = compute_positive_extrema_bisection(Model, T, epss, drift)
#     return t_opt
# end

# @btime compute_extrema_bisection_full_from_param($sigma, $lam, $mu1, $sigma1, $S0, $r, $d, $K, $T, $epss);
# using Zygote, FiniteDiff
# @show Zygote.gradient(compute_extrema_bisection_full_from_param, sigma, lam, mu1, sigma1, S0, r, d, K, T, epss)
# @show FiniteDiff.finite_difference_gradient(x -> compute_extrema_bisection_full_from_param(x..., epss), [sigma, lam, mu1, sigma1, S0, r, d, K, T])
# @btime Zygote.gradient(compute_extrema_bisection_full_from_param, $sigma, $lam, $mu1, $sigma1, $S0, $r, $d, $K, $T, $epss)
