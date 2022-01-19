using DualNumbers, FinancialFFT, FinancialMonteCarlo, BenchmarkTools

A = 400.0;
N = 14;

method_cm = CarrMadanMethod(A, N);
method_lewis = LewisMethod(A, 2^N);
method_cm_lewis = CarrMadanLewisMethod(A, 2^N);
S0 = 100.0;
K = 101.0;
r = dual(0.02, 1.0);
#r = 0.02;
T = 1.0;
d = 0.0;
#sigma=dual(0.2,1.0);
sigma = 0.2;
lam = 15.0;
mu1 = 0.03;
sigma1 = 0.02;
zero_rate = ZeroRate(r);

#Model=MertonProcess(sigma,lam,mu1,sigma1,Underlying(S0,d));
Model = BlackScholesProcess(sigma, Underlying(S0, d));

EUData = EuropeanOption(T, K)
Nsim = 10000;
Nstep = 30;
mc = MonteCarloConfiguration(Nsim, Nstep);

@btime pricer(Model, zero_rate, method_cm, EUData);
@btime pricer(Model, zero_rate, method_lewis, EUData);
@btime pricer(Model, zero_rate, method_cm_lewis, EUData);
