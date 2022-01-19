using FinancialToolbox, DualNumbers, HyperDualNumbers, FinancialFFT, FinancialMonteCarlo

A = 600.0;
N = 18;

method_cm = CarrMadanMethod(A, N);
S0 = 100.0
K = 100.0;
r = hyper(0.02, 1.0, 1.0, 0.0);

T = 1.0;
d = 0.01
sigma = 0.2
lam = 15.0;
mu1 = 0.03;
sigma1 = 0.02;
zero_rate = ZeroRate(r);

Model = VarianceGammaProcess(sigma, 0.03, 0.01, Underlying(S0, d))

EUData = EuropeanOption(T, K)
Nsim = 100000;
Nstep = 30;
function test_hyper(x, y, toll)
    @test abs(x.value - y.value) < toll
    @test abs(x.epsilon1 - y.epsilon1) < toll
    @test abs(x.epsilon2 - y.epsilon2) < toll
    @test abs(x.epsilon12 - y.epsilon12) < toll
    nothing
end

result_cm = pricer(Model, zero_rate, method_cm, EUData);
method_lewis = LewisMethod(700.0, 200000);
analytic_result = pricer(Model, zero_rate, method_lewis, EUData);
method_cm_lewis = CarrMadanLewisMethod(A, N);
result_cm_lewis = pricer(Model, zero_rate, method_cm_lewis, EUData);
toll = 1e-3;
#test_hyper(analytic_result, result_cm, toll)
test_hyper(analytic_result, result_cm_lewis, toll)