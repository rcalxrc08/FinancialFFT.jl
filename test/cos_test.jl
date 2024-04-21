# Option Parameters
using FinancialMonteCarlo, FinancialToolbox, FinancialFFT
S0 = 100.0
r = 0.01
d = 0.0
T = 1.1
sigma = 0.2
K = 120.0
N = 2^13
Model = BlackScholesProcess(sigma, Underlying(S0, d));
opt = EuropeanOption(T, K)
method = CosMethod(N)
z_r = ZeroRate(r)
@show blsprice(S0, K, r, T, sigma, d)
@show blsbin(S0, K, r, T, sigma, d)
@show FinancialFFT.cos_method_pricer(Model, z_r, method, opt, T)
@show FinancialFFT.cos_method_pricer(Model, z_r, method, BinaryEuropeanOption(T, K), T)
# using BenchmarkTools
KK = collect(100.0 * (1:20) / 10.0)
opts = EuropeanOption.(T, KK)
# @show FinancialFFT.cos_method_pricer(Model, z_r, method, opts, T);
# using DualNumbers
# @show FinancialFFT.cos_method_pricer(Model, z_r, method, EuropeanOption(T, dual(K, 1.0)), T)
# using CUDA
# cuda_mode = FinancialMonteCarlo.CudaMode()
# CUDA.allowscalar(false)
@show FinancialFFT.pricer(Model, z_r, method, opt);
@show FinancialFFT.pricer(Model, z_r, method, opts);
# @btime FinancialFFT.cos_method_pricer($Model, $z_r, $method, $opt, $T);
# @btime FinancialFFT.cos_method_pricer($Model, $z_r, $method, $opt, $T, $cuda_mode);

# @btime FinancialFFT.pricer($Model, $z_r, $method, $opt);
# @btime FinancialFFT.pricer($Model, $z_r, $method, $opt, $cuda_mode);

# @btime FinancialFFT.pricer($Model, $z_r, $method, $opts);
# @btime FinancialFFT.pricer($Model, $z_r, $method, $opts, $cuda_mode);

# function blsprice_cos(S0, K, r, T, sigma, d)
#     Model = BlackScholesProcess(sigma, Underlying(S0, d))
#     opt = EuropeanOption(T, K)
#     N = 2^13
#     method = CosMethod(N)
#     z_r = ZeroRate(r)
#     return FinancialFFT.cos_method_pricer(Model, z_r, method, opt, T)
# end
# function blsprice_cos_cu(S0, K, r, T, sigma, d)
#     Model = BlackScholesProcess(sigma, Underlying(S0, d))
#     opt = EuropeanOption(T, K)
#     N = 2^13
#     method = CosMethod(N)
#     cuda_mode = FinancialMonteCarlo.CudaMode()
#     z_r = ZeroRate(r)
#     return FinancialFFT.cos_method_pricer(Model, z_r, method, opt, T, cuda_mode)
# end
# function blsprice_cos_cu(x)
#     return blsprice_cos_cu(x...)
# end
# function blsprice_cos(x)
#     return blsprice_cos(x...)
# end
# using Zygote
# @show "Zygote"
# @btime Zygote.gradient(blsprice_cos, $S0, $K, $r, $T, $sigma, $d);
# @btime Zygote.gradient(blsprice_cos_cu, $S0, $K, $r, $T, $sigma, $d);

# using ForwardDiff
# @show "ForwardDiff"
# inputs_fwd_diff = [S0, K, r, T, sigma, d]
# @btime ForwardDiff.gradient(blsprice_cos, $inputs_fwd_diff);
# @btime ForwardDiff.gradient(blsprice_cos_cu, $inputs_fwd_diff);