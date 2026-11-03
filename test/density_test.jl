# The density interface, as used outside of the pricers (Vibrato, likelihood ratio method).
@testset "Densities" begin
    S0, r, T, σ, d = 100.0, 0.02, 1.1, 0.2, 0.01
    # exact density of log(S_T / S0) in Black Scholes
    lognormal(y, σ) = exp(-(y - (r - d - σ^2 / 2) * T)^2 / (2 * σ^2 * T)) / (σ * sqrt(2 * pi * T))
    model(σ) = BlackScholesProcess(σ, Underlying(S0, d))
    @testset "$name" for (name, method) in ("CosDensityMethod" => CosDensityMethod(), "DensityInverter" => FinancialFFT.DensityInverter(14))
        f = density(model(σ), ZeroRate(r), method, T)
        @test f isa FinancialFFT.AbstractDensity
        y = collect(range(-6σ, 6σ, length = 101))
        @test maximum(abs.(f.(y) .- lognormal.(y, σ))) < 1e-13
        # beyond the range the density is zero
        @test iszero(f(f.b + 0.1)) && iszero(f(f.a - 0.1))
        # it does not depend on S0
        @test f(0.1) == density(BlackScholesProcess(σ, Underlying(2 * S0, d)), ZeroRate(r), method, T)(0.1)
        @testset "Score ∂σ log f at $k σ" for k in (-3, 0, 3)
            score = ForwardDiff.derivative(s -> log(density(model(s), ZeroRate(r), method, T)(k * σ)), σ)
            @test score ≈ ForwardDiff.derivative(s -> log(lognormal(k * σ, s)), σ) rtol = 1e-10
        end
        @test ForwardDiff.derivative(f, 0.1) ≈ ForwardDiff.derivative(y -> lognormal(y, σ), 0.1) rtol = 1e-10
    end
    # the Clenshaw recurrence of a point and the broadcast of the pricer
    f = density(model(σ), ZeroRate(r), CosDensityMethod(), T)
    y = collect(range(-6σ, 6σ, length = 101))
    @test maximum(abs.(f.(y) .- FinancialFFT.cos_density_values(f, y))) < 1e-13
    @testset "$name integrates to one" for (name, process) in model_cases(S0, d)
        for f in (density(process, ZeroRate(r), CosDensityMethod(), T), density(process, ZeroRate(r), FinancialFFT.DensityInverter(14, 5.0), T))
            y = range(f.a, f.b, length = 20001)
            @test sum(f.(y)) * step(y) ≈ 1 atol = 1e-8
        end
    end
    @test_throws MethodError density(model(σ), ZeroRate(r), LewisMethod(600.0, 20000), T)
end
