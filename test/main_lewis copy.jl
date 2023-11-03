using FinancialToolbox, HyperDualNumbers, DualNumbers, FinancialFFT, FinancialMonteCarlo, BenchmarkTools

A = 600.0;
N = 20000;

method = LewisMethod(A, N);
S0 = 100.0;
# S0 = hyper(100.0, 1.0, 1.0, 0.0);
K = 100.0;
# r = hyper(0.02, 1.0, 1.0, 0.0);
r = 0.02;
T = 1.0;
d = 0.0;
#sigma=dual(0.2,1.0);
sigma = 0.2;
# sigma = hyper(0.2, 1.0, 1.0, 0.0);
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
zero_rate = ZeroRate(r);

Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
# Model = KouProcess(sigma, lam, p, lamp, lamm, Underlying(S0, d));
#Model = BlackScholesProcess(sigma, Underlying(S0, d));
# Model = VarianceGammaProcess(sigma, 0.01, 0.01, Underlying(S0, d));
#Model = NormalInverseGaussianProcess(sigma, 0.01, 0.01, Underlying(S0, d));

EUData = EuropeanOption(T, K)
Nsim = 10_000;
Nstep = 30;

@btime pricer($Model, $zero_rate, $method, $EUData);
