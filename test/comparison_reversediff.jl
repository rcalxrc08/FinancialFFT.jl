using FinancialToolbox, Test, FinancialFFT, FinancialMonteCarlo, ReverseDiff

A = 600.0;
N = 18;

S0 = 100.0;
K = 100.0;
r = 0.02;
T = 1.0;
d = 0.01
sigma = 0.2;
lam = 15.0;
mu1 = 0.03;
sigma1 = 0.02;
zero_rate = ZeroRate(r);

Model = BlackScholesProcess(sigma, Underlying(S0, d));

EUData = EuropeanOption(T, K)
Nsim = 100000;
Nstep = 30;
mc = MonteCarloConfiguration(Nsim, Nstep, FinancialMonteCarlo.AntitheticMC());
analytic_result = blsprice(S0, K, r, T, sigma, d);
function test_reverse_diff(x, y, toll)
    @test abs(sum(x .- y)) < toll
    nothing
end
x = Float64[sigma, S0, r]
method_cm = CarrMadanMethod(A, N);
method_lewis = LewisMethod(700.0, 200000);
method_cm_lewis = CarrMadanLewisMethod(A, N);
f(x) = blsprice(x[2], K, x[3], T, x[1], d);
f_lewis(x) = pricer(BlackScholesProcess(x[1], Underlying(x[2], d)), ZeroRate(x[3]), method_lewis, EUData);
analytic_result = ReverseDiff.gradient(f, x);
@show result_lewis = ReverseDiff.gradient(f_lewis, x);
toll = 1e-2;
@testset "ReverseDiff test" begin
    test_reverse_diff(analytic_result, result_lewis, toll)
end