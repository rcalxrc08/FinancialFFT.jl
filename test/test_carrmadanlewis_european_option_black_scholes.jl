using FinancialToolbox, HyperDualNumbers, DualNumbers, FinancialFFT, FinancialMonteCarlo, Zygote, Test

A = 600.0;
const N_lewis = 16;

const method_c = CarrMadanLewisMethod(A, N_lewis);
S0 = 100.0;
# S0 = hyper(100.0, 1.0, 1.0, 0.0);
K = 103.0;
# r = hyper(0.02, 1.0, 1.0, 0.0);
r = 0.02;
T = 1.0;
d = 0.0;
#sigma=dual(0.2,1.0);
sigma = 0.2;
# sigma = hyper(0.2, 1.0, 1.0, 0.0);
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
zero_rate = ZeroRate(r);

Model = BlackScholesProcess(sigma, Underlying(S0, d));

function blsprice_carrmadanlewis(S0, K, r, T, sigma, d, A)
    price = pricer(BlackScholesProcess(sigma, Underlying(S0, d)), ZeroRate(r), CarrMadanLewisMethod(A, N_lewis), EuropeanOption(T, K))
    return price
end
@show blsprice(S0, K, r, T, sigma, d)
@show blsprice_carrmadanlewis(S0, K, r, T, sigma, d, A)
@show Zygote.gradient(blsprice, S0, K, r, T, sigma, d)
@show Zygote.gradient(blsprice_carrmadanlewis, S0, K, r, T, sigma, d, A)

# function blsbin_lewis(S0, K, r, T, sigma, d, A)
#     price = pricer(BlackScholesProcess(sigma, Underlying(S0, d)), ZeroRate(r), CarrMadanLewisMethod(A, N_lewis), BinaryEuropeanOption(T, K))
#     return price
# end

# @show Zygote.gradient(blsbin, S0, K, r, T, sigma, d)
# @show Zygote.gradient(blsbin_lewis, S0, K, r, T, sigma, d, A)

toll = 1e-3

res_analytic_eu = Zygote.gradient(blsprice, S0, K, r, T, sigma, d)
res_analytic_eu_lewis = Zygote.gradient(blsprice_carrmadanlewis, S0, K, r, T, sigma, d, A)

# for i in eachindex(res_analytic_eu)
#     @test abs(res_analytic_eu[i] - res_analytic_eu_lewis[i]) < toll
# end

# res_analytic_bin = Zygote.gradient(blsbin, S0, K, r, T, sigma, d)
# res_analytic_bin_lewis = Zygote.gradient(blsbin_lewis, S0, K, r, T, sigma, d, A)

# for i in eachindex(res_analytic_eu)
#     @test abs(res_analytic_bin[i] - res_analytic_bin_lewis[i]) < toll
# end

sigma_d = dual(sigma, 1.0)
res_analytic_eu = blsprice(S0, K, r, T, sigma_d, d)
res_analytic_eu_lewis = blsprice_carrmadanlewis(S0, K, r, T, sigma_d, d, A)
@test abs(res_analytic_eu.value - res_analytic_eu_lewis.value) < toll
@test abs(res_analytic_eu.epsilon - res_analytic_eu_lewis.epsilon) < toll

# res_analytic_bin = blsbin(S0, K, r, T, sigma_d, d)
# res_analytic_bin_lewis = blsbin_lewis(S0, K, r, T, sigma_d, d, A)
# @test abs(res_analytic_bin.value - res_analytic_bin_lewis.value) < toll
# @test abs(res_analytic_bin.epsilon - res_analytic_bin_lewis.epsilon) < toll

sigma_h = hyper(sigma, 1.0, 1.0, 0.0)
res_analytic_eu = blsprice(S0, K, r, T, sigma_h, d)
res_analytic_eu_lewis = blsprice_carrmadanlewis(S0, K, r, T, sigma_h, d, A)
@test abs(res_analytic_eu.value - res_analytic_eu_lewis.value) < toll
@test abs(res_analytic_eu.epsilon1 - res_analytic_eu_lewis.epsilon1) < toll
@test abs(res_analytic_eu.epsilon2 - res_analytic_eu_lewis.epsilon2) < toll
@test abs(res_analytic_eu.epsilon12 - res_analytic_eu_lewis.epsilon12) < toll

# res_analytic_bin = blsbin(S0, K, r, T, sigma_h, d)
# res_analytic_bin_lewis = blsbin_lewis(S0, K, r, T, sigma_h, d, A)
# @test abs(res_analytic_bin.value - res_analytic_bin_lewis.value) < toll
# @test abs(res_analytic_bin.epsilon1 - res_analytic_bin_lewis.epsilon1) < toll
# @test abs(res_analytic_bin.epsilon2 - res_analytic_bin_lewis.epsilon2) < toll
# @test abs(res_analytic_bin.epsilon12 - res_analytic_bin_lewis.epsilon12) < toll