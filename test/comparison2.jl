using FinancialToolbox, Test, DualNumbers, FinancialFFT, FinancialMonteCarlo

A = 400.0;
N = 18;

S0 = 100.0;
K = 101.0;
r = 0.02;
T = 1.0;
d = 0.01
sigma = 0.2;
@show analytic_result = blsprice(S0, K, r, T, sigma, d);
function test_method_f(S0, K, r, T, sigma, d, iscall, abs_method, toll)
    @show x = blsprice(S0, K, r, T, sigma, d, iscall)
    @show abs_method
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    zero_rate = ZeroRate(r)
    EUData = EuropeanOption(T, K, iscall)
    @show y = pricer(Model, zero_rate, abs_method, EUData)
    descr = string(typeof(abs_method)) * " _ " * string(iscall)
    @testset "$descr" begin
        @test abs(x.value - y.value) < toll
        @test abs(x.epsilon - y.epsilon) < toll
    end
    nothing
end

toll = 1e-2;

method_lewis = LewisMethod(700.0, 200000);
method_cm_lewis = CarrMadanLewisMethod(A, N);
method_density = FinancialFFT.DensityInverter(18)
iscall = true
test_method_f(S0, K, r, T, dual(sigma, 1.0), d, iscall, method_lewis, toll)
test_method_f(S0, K, r, T, dual(sigma, 1.0), d, iscall, method_cm_lewis, toll)
test_method_f(S0, K, r, T, dual(sigma, 1.0), d, iscall, method_density, toll)
iscall = false
test_method_f(S0, K, r, T, dual(sigma, 1.0), d, iscall, method_lewis, toll)
test_method_f(S0, K, r, T, dual(sigma, 1.0), d, iscall, method_cm_lewis, toll)
test_method_f(S0, K, r, T, dual(sigma, 1.0), d, iscall, method_density, toll)

iscall = true
test_method_f(S0, K, r, dual(T, 1.0), sigma, d, iscall, method_lewis, toll)
test_method_f(S0, K, r, dual(T, 1.0), sigma, d, iscall, method_cm_lewis, toll)
test_method_f(S0, K, r, dual(T, 1.0), sigma, d, iscall, method_density, toll)
iscall = false
test_method_f(S0, K, r, dual(T, 1.0), sigma, d, iscall, method_lewis, toll)
test_method_f(S0, K, r, dual(T, 1.0), sigma, d, iscall, method_cm_lewis, toll) #Good catch
test_method_f(S0, K, r, dual(T, 1.0), sigma, d, iscall, method_density, toll)

iscall = true
test_method_f(S0, K, r, T, sigma, dual(d, 1.0), iscall, method_lewis, toll)
test_method_f(S0, K, r, T, sigma, dual(d, 1.0), iscall, method_cm_lewis, toll)
test_method_f(S0, K, r, T, sigma, dual(d, 1.0), iscall, method_density, toll)
iscall = false
test_method_f(S0, K, r, T, sigma, dual(d, 1.0), iscall, method_lewis, toll)
test_method_f(S0, K, r, T, sigma, dual(d, 1.0), iscall, method_cm_lewis, toll)#Good catch
test_method_f(S0, K, r, T, sigma, dual(d, 1.0), iscall, method_density, toll)

iscall = true
test_method_f(S0, dual(K, 2.0), r, T, dual(sigma, 1.0), d, iscall, method_lewis, toll)
test_method_f(S0, dual(K, 2.0), r, T, dual(sigma, 1.0), d, iscall, method_cm_lewis, toll)
test_method_f(S0, dual(K, 2.0), r, T, dual(sigma, 1.0), d, iscall, method_density, toll)
iscall = false
test_method_f(S0, dual(K, 2.0), r, T, dual(sigma, 1.0), d, iscall, method_lewis, toll)
test_method_f(S0, dual(K, 2.0), r, T, dual(sigma, 1.0), d, iscall, method_cm_lewis, toll)#Good catch
test_method_f(S0, dual(K, 2.0), r, T, dual(sigma, 1.0), d, iscall, method_density, toll)