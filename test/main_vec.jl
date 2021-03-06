using FinancialToolbox, DualNumbers, FinancialFFT, FinancialMonteCarlo

A = 600.0;
N = 16;

method = CarrMadanMethod(A, N);
method_cml = CarrMadanLewisMethod(A, N);
method_l = LewisMethod(A, N);
# S0 = 100.0;
S0 = hyper(100.0, 1.0, 1.0, 0.0);
K = 100.0;
r = 0.02;
T = 1.0;
d = 0.0;
sigma = 0.2;
# sigma = hyper(0.2, 1.0, 1.0, 0.0);
# sigma = dual(0.2, 1.0);
lam = 5.0;
mu1 = 0.03;
sigma1 = 0.02;
zero_rate = ZeroRate(r);

Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
# Model = NormalInverseGaussianProcess(sigma, 0.01, 0.01, Underlying(S0, d));
#Model = BlackScholesProcess(sigma, Underlying(S0, d));
options = Array{FinancialMonteCarlo.EuropeanOption}(undef, 10)
options[1:3] = [EuropeanOption(T, K) for i = 1:3];
options[8:10] = [EuropeanOption(T, K) for i = 1:3];
options[4:7] = [EuropeanOption(T, 70.0) for i = 4:7];
Nsim = 10000;
Nstep = 30;
mc = MonteCarloConfiguration(Nsim, Nstep);

@time @show pricer(Model, zero_rate, method, options);
@time @show pricer(Model, zero_rate, method_cml, options);
@time @show pricer(Model, zero_rate, method_l, options);
@time @show pricer(Model, zero_rate, mc, options);
typeof(Model) <: BlackScholesProcess ? @time(@show(blsprice(S0, K, r, T, sigma, d))) : nothing;
