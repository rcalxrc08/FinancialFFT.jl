using FinancialToolbox, FinancialFFT, FinancialMonteCarlo, BenchmarkTools, Zygote, CUDA, Test

A = 600.0;
N = 14;

S0 = 100.0;
K = 104.0;
r = 0.02;
#r=0.02;
T = 1.0;
d = 0.0;
#sigma=dual(0.2,1.0);
sigma = 0.2;
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
zero_rate = ZeroRate(r);

Model = BlackScholesProcess(sigma, Underlying(S0, d));
N = 14
xmax = 10.0
# @btime FinancialFFT.density($Model, $T, $r, $N, $xmax, $Model.underlying.S0);
# @btime FinancialFFT.density_new($Model, $T, $r, $N, $xmax, $Model.underlying.S0);
# p = FinancialFFT.density(Model, T, r, N, xmax, Model.underlying.S0);
function black_density_old(S0, K, r, T, sigma, d)
    opt = BinaryEuropeanOption(T, K)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    method = FinancialFFT.DensityInverter(18)
    zero_rate = FinancialMonteCarlo.ZeroRate(r)
    #mode = FinancialMonteCarlo.SerialMode()
    mode = FinancialMonteCarlo.CudaMode()
    price = pricer(Model, zero_rate, method, opt, mode)
    return price
end
@show black_density_old(S0, K, r, T, sigma, d)
@show blsbin(S0, K, r, T, sigma, d)
@test abs(blsbin(S0, K, r, T, sigma, d) - black_density_old(S0, K, r, T, sigma, d)) < 1e-3
@show Zygote.gradient(black_density_old, S0, K, r, T, sigma, d)
@show Zygote.gradient(blsbin, S0, K, r, T, sigma, d)
@btime black_density_old($S0, $K, $r, $T, $sigma, $d);
@btime Zygote.gradient(black_density_old, $S0, $K, $r, $T, $sigma, $d)