using FinancialToolbox, Test, FinancialFFT, FinancialMonteCarlo, Zygote

A = 400.0;
N = 18;

S0 = 100.0;
K = 111.0;
r = 0.02;
T = 1.0;
d = 0.01
sigma = 0.2;
@show analytic_result = blsbin(S0, K, r, T, sigma, d);
function test_method_f(S0, K, r, T, sigma, d, iscall, abs_method, toll)
    f(S0, K, r, T, sigma, d) = blsbin(S0, K, r, T, sigma, d, iscall)
    @show x = Zygote.gradient(f, S0, K, r, T, sigma, d)
    # g(S0, K, r, T, sigma, d) = pricer(BlackScholesProcess(sigma, Underlying(S0, d)), ZeroRate(r), abs_method, BinaryEuropeanOption(T, K, iscall), FinancialMonteCarlo.CudaMode())
    g(S0, K, r, T, sigma, d) = pricer(BlackScholesProcess(sigma, Underlying(S0, d)), ZeroRate(r), abs_method, EuropeanOption(T, K, iscall))
    @show y = Zygote.gradient(g, S0, K, r, T, sigma, d)
    descr = string(typeof(abs_method)) * " _ " * string(iscall)
    @testset "$descr" begin
        for (xx, yy) in zip(x, y)
            @test abs(xx - yy) < toll
        end
        # @test abs(x.epsilon - y.epsilon) < toll
    end
    nothing
end

toll = 1e-2;

method_lewis = LewisMethod(700.0, 200000);
method_cm_lewis = CarrMadanLewisMethod(A, N);
method_density = FinancialFFT.DensityInverter(18)
iscall = true
test_method_f(S0, K, r, T, sigma, d, iscall, method_lewis, toll)
# test_method_f(S0, K, r, T, sigma, d, iscall, method_cm_lewis, toll)
test_method_f(S0, K, r, T, sigma, d, iscall, method_density, toll)
iscall = false
test_method_f(S0, K, r, T, sigma, d, iscall, method_lewis, toll)
# test_method_f(S0, K, r, T, sigma, d, iscall, method_cm_lewis, toll)
test_method_f(S0, K, r, T, sigma, d, iscall, method_density, toll)
