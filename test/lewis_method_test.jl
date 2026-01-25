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
A = 600.0;
N = 20000;

method = LewisMethod(A, N);
z_r = ZeroRate(r)
function blsprice_lewis(S0, K, r, T, sigma, d, iscall = true)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = EuropeanOption(T, K, iscall)
    A = 600.0
    N = 20000
    method = LewisMethod(A, N)
    z_r = ZeroRate(r)
    return pricer(Model, z_r, method, opt)
end
function blsprice_lewis_smile(S0, K, r, T, sigma, d, iscall = true)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = FinancialFFT.EuropeanOptionSmile(T, [K], [iscall])
    A = 600.0
    N = 20000
    method = LewisMethod(A, N)
    z_r = ZeroRate(r)
    return pricer(Model, z_r, method, opt)[1]
end
function blsbin_lewis(S0, K, r, T, sigma, d, iscall = true)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = BinaryEuropeanOption(T, K, iscall)
    A = 600.0
    N = 20000
    method = LewisMethod(A, N)
    z_r = ZeroRate(r)
    return pricer(Model, z_r, method, opt)
end
toll = 1e-10

@testset "Lewis Method Test" begin
    lewis_price = blsprice_lewis(S0, K, r, T, sigma, d)
    lewis_price_smile = blsprice_lewis_smile(S0, K, r, T, sigma, d)
    price = blsprice(S0, K, r, T, sigma, d)
    @test abs(lewis_price - price) < toll
    @test abs(lewis_price_smile - price) < toll
    price_lewis_bin = blsbin_lewis(S0, K, r, T, sigma, d)
    price_bin = blsbin(S0, K, r, T, sigma, d)
    @test abs(price_lewis_bin - price_bin) < toll
    lewis_price = blsprice_lewis(S0, K, r, T, sigma, d, false)
    lewis_price_smile = blsprice_lewis_smile(S0, K, r, T, sigma, d, false)
    price = blsprice(S0, K, r, T, sigma, d, false)
    @test abs(lewis_price - price) < toll
    @test abs(lewis_price_smile - price) < toll
    price_lewis_bin = blsbin_lewis(S0, K, r, T, sigma, d, false)
    price_bin = blsbin(S0, K, r, T, sigma, d, false)
    @test abs(price_lewis_bin - price_bin) < toll
end

@testset "Lewis Method HyperDuals Test" begin
    # Option Parameters
    using HyperDualNumbers
    S0 = hyper(100.0, 1.0, 0.0, 0.0)
    sigma = hyper(0.2, 0.0, 1.0, 0.0)
    toll = 1e-10
    lewis_price = blsprice_lewis(S0, K, r, T, sigma, d)
    lewis_price_smile = blsprice_lewis_smile(S0, K, r, T, sigma, d)
    price = blsprice(S0, K, r, T, sigma, d)
    @test abs(lewis_price - price) < toll
    @test abs(lewis_price_smile - price) < toll
    price_lewis_bin = blsbin_lewis(S0, K, r, T, sigma, d)
    price_bin = blsbin(S0, K, r, T, sigma, d)
    @test abs(price_lewis_bin - price_bin) < toll
end

@testset "Lewis Method Zygote Test" begin
    gradient_lewis_price = Zygote.gradient(blsprice_lewis, S0, K, r, T, sigma, d)
    gradient_lewis_price_smile = Zygote.gradient(blsprice_lewis_smile, S0, K, r, T, sigma, d)
    gradient_price = Zygote.gradient(blsprice, S0, K, r, T, sigma, d)
    @test maximum(abs.(gradient_lewis_price .- gradient_price)) < toll
    @test maximum(abs.(gradient_lewis_price_smile .- gradient_price)) < toll
    gradient_lewis_bin = Zygote.gradient(blsbin_lewis, S0, K, r, T, sigma, d)
    gradient_bin = Zygote.gradient(blsbin, S0, K, r, T, sigma, d)
    @test maximum(abs.(gradient_lewis_bin .- gradient_bin)) < toll
end