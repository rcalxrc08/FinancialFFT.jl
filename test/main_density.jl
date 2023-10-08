using FinancialToolbox, FinancialFFT, FinancialMonteCarlo, BenchmarkTools, Zygote

A = 600.0;
N = 14;

S0 = 100.0;
K = 100.0;
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
p = FinancialFFT.density(Model, T, r, N, xmax, Model.underlying.S0);
function black_density(sigma, S0, d, r, T)
    xmax = 10.0
    N = 14
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    density_pair = FinancialFFT.density_new(Model, T, r, N, xmax, Model.underlying.S0)
    return sum(density_pair[2])
end
function black_density_old(sigma, S0, d, r, T, K)
    opt = BinaryEuropeanOption(T, K)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    price = FinancialFFT.pricer_from_density(Model, T, r, opt)
    return price
end
@show black_density_old(sigma, S0, d, r, T, K)
@show blsbin(S0, K, r, T, sigma, d)
@show Zygote.gradient(black_density_old, sigma, S0, d, r, T, K)
@show Zygote.gradient(blsbin, S0, K, r, T, sigma, d)
@btime black_density_old($sigma, $S0, $d, $r, $T, $K);
@btime Zygote.gradient(black_density_old, $sigma, $S0, $d, $r, $T, $K)