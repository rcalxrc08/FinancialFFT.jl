using FinancialToolbox, Test, DualNumbers, HyperDualNumbers, FinancialFFT, FinancialMonteCarlo

A = 400.0;
N = 18;

method_cm = CarrMadanMethod(A, N);
S0 = 100.0;
K = 107.0;
r = 0.02;
T = 1.0;
d = 0.01
sigma = dual(0.2, 1.0);
lam = 15.0;
mu1 = 0.03;
sigma1 = 0.02;
zero_rate = ZeroRate(r);

Model = BlackScholesProcess(sigma, Underlying(S0, d));

EUData = EuropeanOption(T, K)
Nsim = 100000;
Nstep = 30;
mc = MonteCarloConfiguration(Nsim, Nstep, FinancialMonteCarlo.AntitheticMC());
@show analytic_result = blsprice(S0, K, r, T, sigma, d);
function test_dual(x, y, toll)
    @test abs(x.value - y.value) < toll
    @test abs(x.epsilon - y.epsilon) < toll
    nothing
end

@show result_cm = pricer(Model, zero_rate, method_cm, EUData);
method_lewis = LewisMethod(700.0, 200000);
@show result_lewis = pricer(Model, zero_rate, method_lewis, EUData);
method_cm_lewis = CarrMadanLewisMethod(A, N);
@show result_cm_lewis = pricer(Model, zero_rate, method_cm_lewis, EUData);
method_density = FinancialFFT.DensityInverter(18)
@show result_density = pricer(Model, zero_rate, method_density, EUData);
toll = 1e-2;
@testset "dual test" begin
    #test_dual(analytic_result, result_cm, toll)
    test_dual(analytic_result, result_lewis, toll)
    test_dual(analytic_result, result_cm_lewis, toll)
    test_dual(analytic_result, result_density, toll)
end