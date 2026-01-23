using FinancialMonteCarlo, FinancialToolbox, FinancialFFT, Test, Zygote

# Option Parameters
S0 = 100.0
r = 0.01
d = 0.0
T = 1.1
sigma = 0.2
K = 120.0
N = 2^7
Model = BlackScholesProcess(sigma, Underlying(S0, d))
opt = EuropeanOption(T, K)
method = CosMethod(N)
z_r = ZeroRate(r)
function blsprice_cos(S0, K, r, T, sigma, d, iscall = true)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = EuropeanOption(T, K, iscall)
    N = 2^12
    method = CosMethod(N)
    z_r = ZeroRate(r)
    return pricer(Model, z_r, method, opt)
end
function blsprice_cos_smile(S0, K, r, T, sigma, d, iscall = true)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = FinancialFFT.EuropeanOptionSmile(T, [K], [iscall])
    N = 2^12
    method = CosMethod(N)
    z_r = ZeroRate(r)
    return pricer(Model, z_r, method, opt)[1]
end
function blsbin_cos(S0, K, r, T, sigma, d, iscall = true)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = BinaryEuropeanOption(T, K, iscall)
    N = 2^12
    method = CosMethod(N)
    z_r = ZeroRate(r)
    return pricer(Model, z_r, method, opt)
end
toll = 1e-10

@testset "Cos Method Test" begin
    cos_price = blsprice_cos(S0, K, r, T, sigma, d)
    cos_price_smile = blsprice_cos_smile(S0, K, r, T, sigma, d)
    price = blsprice(S0, K, r, T, sigma, d)
    @test abs(cos_price - price) < toll
    @test abs(cos_price_smile - price) < toll
    price_cos_bin = blsbin_cos(S0, K, r, T, sigma, d)
    price_bin = blsbin(S0, K, r, T, sigma, d)
    @test abs(price_cos_bin - price_bin) < toll

    cos_price = blsprice_cos(S0, K, r, T, sigma, d, false)
    cos_price_smile = blsprice_cos_smile(S0, K, r, T, sigma, d, false)
    price = blsprice(S0, K, r, T, sigma, d, false)
    @test abs(cos_price - price) < toll
    @test abs(cos_price_smile - price) < toll
    price_cos_bin = blsbin_cos(S0, K, r, T, sigma, d, false)
    price_bin = blsbin(S0, K, r, T, sigma, d, false)
    @test abs(price_cos_bin - price_bin) < toll
end

@testset "Cos Method HyperDuals Test" begin
    # Option Parameters
    using HyperDualNumbers
    S0 = hyper(100.0, 1.0, 0.0, 0.0)
    sigma = hyper(0.2, 0.0, 1.0, 0.0)
    toll = 1e-10
    cos_price = blsprice_cos(S0, K, r, T, sigma, d)
    cos_price_smile = blsprice_cos_smile(S0, K, r, T, sigma, d)
    price = blsprice(S0, K, r, T, sigma, d)
    @test abs(cos_price - price) < toll
    @test abs(cos_price_smile - price) < toll
    price_cos_bin = blsbin_cos(S0, K, r, T, sigma, d)
    price_bin = blsbin(S0, K, r, T, sigma, d)
    @test abs(price_cos_bin - price_bin) < toll
end

@testset "Cos Method Zygote Test" begin
    gradient_cos_price = Zygote.gradient(blsprice_cos, S0, K, r, T, sigma, d)
    gradient_price = Zygote.gradient(blsprice, S0, K, r, T, sigma, d)
    @test maximum(abs.(gradient_cos_price .- gradient_price)) < toll
    gradient_cos_bin = Zygote.gradient(blsbin_cos, S0, K, r, T, sigma, d)
    gradient_bin = Zygote.gradient(blsbin, S0, K, r, T, sigma, d)
    @test maximum(abs.(gradient_cos_bin .- gradient_bin)) < toll
end