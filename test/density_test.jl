# The density of cos_density, as used outside of the pricers (Vibrato, likelihood ratio method).
@testset "COS density" begin
    S0, r, T, σ, d = 100.0, 0.02, 1.1, 0.2, 0.01
    # exact density of log(S_T / S0) in Black Scholes
    lognormal(y, σ) = exp(-(y - (r - d - σ^2 / 2) * T)^2 / (2 * σ^2 * T)) / (σ * sqrt(2 * pi * T))
    model(σ) = BlackScholesProcess(σ, Underlying(S0, d))
    density = cos_density(model(σ), ZeroRate(r), T)
    y = collect(range(-6σ, 6σ, length = 101))
    @test maximum(abs.(density.(y) .- lognormal.(y, σ))) < 1e-13
    # the Clenshaw recurrence of a point and the broadcast of the pricers
    @test maximum(abs.(density.(y) .- FinancialFFT.cos_density_values(density, y))) < 1e-13
    # beyond the range the series is periodic: the density is zero there
    @test iszero(density(density.b + 0.1)) && iszero(density(density.a - 0.1))
    # it does not depend on S0
    @test density(0.1) == cos_density(BlackScholesProcess(σ, Underlying(2 * S0, d)), ZeroRate(r), T)(0.1)
    @testset "Score ∂σ log f at $k σ" for k in (-3, 0, 3)
        score = ForwardDiff.derivative(s -> log(cos_density(model(s), ZeroRate(r), T)(k * σ)), σ)
        @test score ≈ ForwardDiff.derivative(s -> log(lognormal(k * σ, s)), σ) rtol = 1e-10
    end
    @test ForwardDiff.derivative(density, 0.1) ≈ ForwardDiff.derivative(y -> lognormal(y, σ), 0.1) rtol = 1e-10
    @testset "$name integrates to one" for (name, process) in model_cases(S0, d)
        density = cos_density(process, ZeroRate(r), T)
        y = range(density.a, density.b, length = 20001)
        @test sum(density.(y)) * step(y) ≈ 1 atol = 1e-8
    end
end
