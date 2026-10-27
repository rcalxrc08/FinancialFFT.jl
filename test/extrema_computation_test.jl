# Truncation range of the COS method, computed from a Chernoff bound on the tails of the log-price.
@testset "Chernoff extrema" begin
    epss = 1e-12
    S0, r, d, K = 100.0, 0.02, 0.01, 100.0
    @testset "Black Scholes closed formula, T = $T" for T in (0.05, 0.5, 1.1, 5.0)
        sigma = 0.2
        model = BlackScholesProcess(sigma, Underlying(S0, d))
        drift = (r - d) * T + log(S0 / K)
        driftT = drift - FinancialFFT.characteristic_exponent_i(1, model) * T
        # P(X - driftT > x) ≤ exp(σ²Ts²/2 - sx), minimized over s and set equal to epss
        half_width = sigma * sqrt(-2 * T * log(epss))
        a, b = FinancialFFT.compute_extrema_bisection_with_default(model, T, epss, drift)
        @test abs(a - (driftT - half_width)) < 1e-8
        @test abs(b - (driftT + half_width)) < 1e-8
        @test FinancialFFT.compute_positive_extrema_bisection(model, T, epss, drift) ≈ b
        @test FinancialFFT.compute_negative_extrema_bisection(model, T, epss, drift) ≈ a
    end
    @testset "$name, T = $T" for (name, model) in model_cases(S0, d), T in (0.01, 0.05, 1.1)
        drift = (r - d) * T + log(S0 / K)
        a, b = FinancialFFT.compute_extrema_bisection_with_default(model, T, epss, drift)
        @test a < drift < b
        @test FinancialFFT.compute_positive_extrema_bisection(model, T, epss, drift) ≈ b
        @test FinancialFFT.compute_negative_extrema_bisection(model, T, epss, drift) ≈ a
        # the optimum is admissible (below the exponential moment bound) and matches a grid search over the admissible s
        @testset "mul = $mul" for mul in (-1, 1)
            log_epss = log(epss)
            bound = FinancialFFT.exponential_moment_bound(model, mul)
            t_opt = -mul * FinancialFFT.compute_chernoff_extrema_bisection_single(model, T, log_epss, 1.0, 50.0, 1e-14, mul, 0.0)
            grid = exp.(range(log(1e-2), log(min(bound, 1e4) * (1 - 1e-9)), length = 10^5))
            t_grid = minimum(s -> FinancialFFT.compute_ts(s, model, T, log_epss, mul), grid)
            @test t_opt > 0
            @test t_opt <= t_grid * (1 + 1e-6)
        end
    end
    @testset "Exponential moment bounds" begin
        @test FinancialFFT.exponential_moment_bound(BlackScholesProcess(0.2, Underlying(S0, d)), 1) == Inf
        kou = KouProcess(0.2, 5.0, 0.3, 30.0, 20.0, Underlying(S0, d))
        @test FinancialFFT.exponential_moment_bound(kou, 1) == 30.0
        @test FinancialFFT.exponential_moment_bound(kou, -1) == 20.0
        # the argument of the log (VG) and of the sqrt (NIG) vanishes at the bound
        for mul in (-1, 1)
            σ, θ, κ = 0.2, 0.03, 0.01
            s = mul * FinancialFFT.exponential_moment_bound(VarianceGammaProcess(σ, θ, κ, Underlying(S0, d)), mul)
            @test abs(1 - s * (s * σ^2 * κ / 2 + θ * κ)) < 1e-12
            s = mul * FinancialFFT.exponential_moment_bound(NormalInverseGaussianProcess(σ, θ, κ, Underlying(S0, d)), mul)
            @test abs(1 - s * (s * σ^2 * κ + 2 * θ * κ)) < 1e-12
        end
    end
end
