# Truncation range of the COS method, computed from a Chernoff bound on the tails of the log-price.

# An exponent that counts its evaluations, to bound the cost of the search.
mutable struct CountingExponent{E}
    exponent::E
    evaluations::Int
end
(e::CountingExponent)(s) = (e.evaluations += 1; e.exponent(s))
FinancialFFT.complex_step(s, e::CountingExponent) = FinancialFFT.complex_step(s, e.exponent)
FinancialFFT.exponential_moment_bound(e::CountingExponent) = FinancialFFT.exponential_moment_bound(e.exponent)

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
        a, b = FinancialFFT.compute_chernoff_extrema(model, T, epss, drift)
        @test abs(a - (driftT - half_width)) < 1e-8
        @test abs(b - (driftT + half_width)) < 1e-8
    end
    # Merton with large jumps: the Gaussian guess from s = 1 overflows the exponent at short maturities
    @testset "$name, T = $T" for (name, model) in vcat(model_cases(S0, d), ["Merton, large jumps" => MertonProcess(0.2, 5.0, 0.0, 0.1, Underlying(S0, d))]), T in (0.001, 0.01, 0.05, 1.1)
        drift = (r - d) * T + log(S0 / K)
        a, b = FinancialFFT.compute_chernoff_extrema(model, T, epss, drift)
        @test a < drift < b
        # the optimum is admissible (below the exponential moment bound) and matches a grid search over the admissible s
        @testset "mul = $mul" for mul in (-1, 1)
            log_epss = log(epss)
            bound = FinancialFFT.exponential_moment_bound(model, mul, T)
            t_opt = mul * FinancialFFT.compute_chernoff_extrema_single(model, T, log_epss, 1.0, sqrt(eps()), mul, 0.0)
            grid = exp.(range(log(1e-2), log(min(bound, 1e4) * (1 - 1e-9)), length = 10^5))
            t_grid = minimum(s -> FinancialFFT.compute_ts(s, model, T, log_epss, mul), grid)
            @test t_opt > 0
            @test t_opt <= t_grid * (1 + 1e-6)
        end
    end
    # Issue #38: Brent's method to a relative tolerance sqrt(eps) on s gives the endpoints of a solve to 1e-14, since t is
    # flat at its minimum, and any s gives a valid bound: never narrower than the tight one beyond rounding.
    @testset "Tolerance and cost of the search, $name, T = $T" for (name, model) in vcat(["BlackScholes" => BlackScholesProcess(0.2, Underlying(S0, d))], model_cases(S0, d)), T in (1e-3, 0.01, 0.25, 1.0, 10.0)
        a, b = FinancialFFT.compute_chernorff_limits(model, r * T, d * T, T)
        epss_range = FinancialFFT.scaled_tolerance(Float64, 1e-14)
        a_tight, b_tight = FinancialFFT.compute_chernoff_extrema(model, T, epss_range, (r - d) * T, 1.0, 1e-14)
        @test abs(a - a_tight) <= 1e-10 * abs(a_tight)
        @test abs(b - b_tight) <= 1e-10 * abs(b_tight)
        @test a <= a_tight + 1e-12 * abs(a_tight)
        @test b >= b_tight - 1e-12 * abs(b_tight)
        # the bisection it replaced took 102-115 evaluations of the exponent per range
        evaluations = sum((-1, 1)) do mul
            exponent = CountingExponent(FinancialFFT.TailExponent(model, T, mul), 0)
            FinancialFFT.chernoff_optimum(exponent, log(epss_range), 1.0, sqrt(eps()))
            exponent.evaluations
        end
        @test evaluations <= 30
    end
    # The wider side of the range is the heavier tail. The models above are too symmetric to tell the two endpoints apart.
    @testset "Skewed models, $name" for (name, model) in ("VarianceGamma" => VarianceGammaProcess(0.12, -0.3, 0.5, Underlying(S0, d)), "Kou" => KouProcess(0.2, 5.0, 0.1, 50.0, 3.0, Underlying(S0, d)), "Heston" => HestonProcess(1.0, 0.2, 0.0, 1.5, -0.9, 0.04, Underlying(S0, d)))
        T = 1.0
        driftT = -FinancialFFT.characteristic_exponent_i(1, model, T)
        a, b = FinancialFFT.compute_chernoff_extrema(model, T, epss, 0.0)
        @test driftT - a > b - driftT
        # each endpoint is the tightest of the bounds exp(T ψ(mul * s) - s * t) on its own tail, with mul = 1 the upper one
        for (mul, x) in ((-1, a), (1, b)), s in (0.1, 0.5, 0.99) .* min(FinancialFFT.exponential_moment_bound(model, mul, T), 1.0)
            @test FinancialFFT.characteristic_exponent_i(mul * s, model, T) - s * mul * (x - driftT) >= log(epss) - 1e-9
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
        # the moments of Heston explode at the maturity T when s is at the bound, and the bound shrinks as T grows
        σ, σ₀, λ, κ, ρ, θ = 0.3, 0.2, 0.0, 1.5, -0.7, 0.04
        heston = HestonProcess(σ, σ₀, λ, κ, ρ, θ, Underlying(S0, d))
        @testset "Heston, mul = $mul, T = $T" for mul in (-1, 1), T in (0.5, 2.0)
            s = FinancialFFT.exponential_moment_bound(heston, mul, T)
            @test FinancialFFT.heston_explosion_rate(mul * s, κ + λ, ρ * σ, σ^2) ≈ 1 / T rtol = 1e-8
            @test FinancialFFT.exponential_moment_bound(heston, mul, 2 * T) < s
            @test isfinite(FinancialFFT.characteristic_exponent_i(mul * s * (1 - 1e-6), heston, T))
        end
        # with ρ = 1 the discriminant stays positive: the upper moments explode and the lower ones never do
        σ, κ, ρ = 1.0, 0.5, 1.0
        perfect_correlation = HestonProcess(σ, σ₀, 0.0, κ, ρ, θ, Underlying(S0, d))
        s = FinancialFFT.exponential_moment_bound(perfect_correlation, 1, 0.5)
        @test FinancialFFT.heston_explosion_rate(s, κ, ρ * σ, σ^2) ≈ 2 rtol = 1e-8
        @test FinancialFFT.exponential_moment_bound(perfect_correlation, -1, 0.5) == Inf
    end
end
