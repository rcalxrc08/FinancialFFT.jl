using FinancialToolbox, DualNumbers, HyperDualNumbers, FinancialFFT, FinancialMonteCarlo

A = 600.0;
N = 18;

method_cm = CarrMadanMethod(A, N);
S0 = 100.0;
# S0 = dual(100.0, 1.0);
K = 100.0;
# K = dual(100.0, 1.0);
# r = hyper(0.02, 1.0, 1.0, 0.0);
# r = dual(0.02, 1.0);
r = 0.02;
T = 1.0;
# T = dual(1.0, 1.0);
# d = hyper(0.0, 0.0, 1.0, 0.0);
d = 0.01
# d = 0.0;
# sigma = dual(0.2, 1.0);
sigma = hyper(0.2, 1.0, 1.0, 0.0);
# sigma = 0.2;
lam = 15.0;
mu1 = 0.03;
sigma1 = 0.02;
zero_rate = ZeroRate(r);

# Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
Model = VarianceGammaProcess(sigma, 0.03, 0.01, Underlying(S0, d))
# Model = BlackScholesProcess(sigma, Underlying(S0, d));

EUData = EuropeanOption(T, K)
Nsim = 10000;
Nstep = 30;
mc = MonteCarloConfiguration(Nsim, Nstep);

@time @show pricer(Model, zero_rate, method_cm, EUData);
method_lewis = LewisMethod(700.0, 200000);
@time @show pricer(Model, zero_rate, method_lewis, EUData);
@time @show pricer(Model, zero_rate, mc, EUData);
method_cm_lewis = CarrMadanLewisMethod(A, N);
@time @show pricer(Model, zero_rate, method_cm_lewis, EUData);