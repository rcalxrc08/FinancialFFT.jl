using FinancialToolbox, FinancialFFT, FinancialMonteCarlo, BenchmarkTools

A = 600.0;
N = 14;

method = CarrMadanMethod(A, N);
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

#Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
#Model = KouProcess(sigma, lam, p, lamp, lamm, Underlying(S0, d));
#Model = BlackScholesProcess(sigma, Underlying(S0, d));
# Model = VarianceGammaProcess(sigma, 0.01, 0.01, Underlying(S0, d));
Model = NormalInverseGaussianProcess(sigma, 0.01, 0.01, Underlying(S0, d));
EUData = EuropeanOption(T, K)
@btime pricer(Model, zero_rate, method, EUData);
