using TaylorSeries,FinancialToolbox, HyperDualNumbers, DualNumbers, FinancialFFT, FinancialMonteCarlo

A = 600.0;
N = 20000;

method = LewisMethod(A, N);
S0 =taylor_expand(identity,100.0,order=5)
# S0 = hyper(100.0, 1.0, 1.0, 0.0);
K = 105.0;
# r = hyper(0.02, 1.0, 1.0, 0.0);
r = 0.02;
# T = 1.0;
T = 1.0
d = 0.0;
#sigma=dual(0.2,1.0);
sigma = 0.2;
# sigma = hyper(0.2, 1.0, 1.0, 0.0);
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 50.0;
lamp = 30.0;
lamm = 20.0;
zero_rate = ZeroRate(r);

# Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
Model = KouProcess(sigma, lam, p, lamp, lamm, Underlying(S0, d));
#Model = BlackScholesProcess(sigma, Underlying(S0, d));
# Model = VarianceGammaProcess(sigma, 0.01, 0.01, Underlying(S0, d));
# Model = HestonProcess(0.2, 0.2, 0.0, 0.01, -0.21, 0.02, Underlying(S0, d));
# Model = NormalInverseGaussianProcess(sigma, 0.01, 1.03, Underlying(S0, d));
# Model = NormalInverseGaussianProcess(sigma, 0.01, 1.03, Underlying(S0, d));

EUData = EuropeanOption(T, K)
Nsim = 10_000;
Nstep = 30;
# mc = MonteCarloConfiguration(Nsim, Nstep, FinancialMonteCarlo.AntitheticMC());
mc = MonteCarloConfiguration(Nsim, Nstep);

@show pricer(Model, zero_rate, method, EUData);
# @show pricer(Model, zero_rate, mc, EUData);
# typeof(Model) <: BlackScholesProcess ? @time(@show(blsprice(S0, K, r, T, sigma, d))) : nothing;
