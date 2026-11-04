# The density interface, as used outside of the pricers (Vibrato, likelihood ratio method).
@testset "Densities" begin
    S0, r, T, σ, d = 100.0, 0.02, 1.1, 0.2, 0.01
    # exact density of log(S_T / S0) in Black Scholes
    lognormal(y, σ) = exp(-(y - (r - d - σ^2 / 2) * T)^2 / (2 * σ^2 * T)) / (σ * sqrt(2 * pi * T))
    model(σ) = BlackScholesProcess(σ, Underlying(S0, d))
    # the cubic interpolation of FFTDensityInverter errs by O(dx⁴) on the values, O(dx³) on the derivative in y
    @testset "$name" for (name, method, toll, rtol_dy) in (("CosDensityMethod", CosDensityMethod(), 1e-13, 1e-10), ("FFTDensityInverter", FFTDensityInverter(), 2e-13, 1e-9))
        f = density(model(σ), ZeroRate(r), method, T)
        @test f isa FinancialFFT.AbstractDensity
        y = collect(range(-6σ, 6σ, length = 101))
        @test maximum(abs.(f.(y) .- lognormal.(y, σ))) < toll
        # beyond the range the density is zero
        @test iszero(f(f.b + 0.1)) && iszero(f(f.a - 0.1))
        # it does not depend on S0
        @test f(0.1) == density(BlackScholesProcess(σ, Underlying(2 * S0, d)), ZeroRate(r), method, T)(0.1)
        @testset "Score ∂σ log f at $k σ" for k in (-3, 0, 3)
            score = ForwardDiff.derivative(s -> log(density(model(s), ZeroRate(r), method, T)(k * σ)), σ)
            @test score ≈ ForwardDiff.derivative(s -> log(lognormal(k * σ, s)), σ) rtol = 1e-10
        end
        @test ForwardDiff.derivative(f, 0.1) ≈ ForwardDiff.derivative(y -> lognormal(y, σ), 0.1) rtol = rtol_dy
    end
    # the Clenshaw recurrence of a point and the broadcast of the pricer
    f = density(model(σ), ZeroRate(r), CosDensityMethod(), T)
    y = collect(range(-6σ, 6σ, length = 101))
    @test maximum(abs.(f.(y) .- FinancialFFT.density_array(f, y))) < 1e-13
    @testset "$name integrates to one" for (name, process) in model_cases(S0, d)
        for f in (density(process, ZeroRate(r), CosDensityMethod(), T), density(process, ZeroRate(r), FFTDensityInverter(), T))
            y = range(f.a, f.b, length = 20001)
            @test sum(f.(y)) * step(y) ≈ 1 atol = 1e-8
        end
    end
    # the FFT density on the range of COS agrees with the COS density on every model, also over a short step, where the
    # range narrows with the standard deviation. Over a short step the Variance Gamma density has a cusp at its mode,
    # which the cosine expansion does not resolve (2% with 512 terms): there the reference is a finer FFT.
    @testset "FFTDensityInverter, $name, T = $τ" for (name, process) in model_cases(S0, d), τ in (T, 0.01)
        f = density(process, ZeroRate(r), FFTDensityInverter(), τ)
        has_cusp = process isa VarianceGammaProcess && τ < T
        g = has_cusp ? density(process, ZeroRate(r), FFTDensityInverter(18), τ) : density(process, ZeroRate(r), CosDensityMethod(512), τ)
        y = range(f.a, f.b, length = 2001)
        @test maximum(abs.(f.(y) .- g.(y))) < (has_cusp ? 1e-3 : 1e-9) * maximum(g.(y))
    end
    @testset "density_values of $name" for (name, method) in ("CosDensityMethod" => CosDensityMethod(), "FFTDensityInverter" => FFTDensityInverter())
        f = density(model(σ), ZeroRate(r), method, T)
        y = [f.a - 0.1, -0.3, 0.0, 0.1, 0.25, f.b + 0.1]
        @test maximum(abs.(FinancialFFT.density_values(f, y) .- f.(y))) < 1e-12
    end
    # Scores of the FFT density with every AD back-end, at a point and at an array of points (gathers, whose pullback
    # is a scatter): the grid is a constant for differentiation in all of them.
    @testset "FFTDensityInverter scores, $backend" for backend in ("DualNumbers", "HyperDualNumbers", "TaylorSeries", "ForwardDiff", "ReverseDiff", "Zygote", "Mooncake")
        k = 3σ
        y = [-2σ, 0.0, k]
        score(s) = log(density(model(s[1]), ZeroRate(r), FFTDensityInverter(), T)(k))
        scores(s) = sum(log.(FinancialFFT.density_values(density(model(s[1]), ZeroRate(r), FFTDensityInverter(), T), y)))
        exact = ForwardDiff.derivative(s -> log(lognormal(k, s)), σ)
        exact_sum = sum(ForwardDiff.derivative(s -> log(lognormal(x, s)), σ) for x in y)
        gradient = Dict("DualNumbers" => h -> DualNumbers.epsilon(h([Dual(σ, 1.0)])), "HyperDualNumbers" => h -> HyperDualNumbers.eps1(h([hyper(σ, 1.0, 1.0, 0.0)])), "TaylorSeries" => h -> TaylorSeries.getcoeff(h([σ + Taylor1(1)]), 1), "ForwardDiff" => h -> ForwardDiff.gradient(h, [σ])[1], "ReverseDiff" => h -> ReverseDiff.gradient(h, [σ])[1], "Zygote" => h -> Zygote.gradient(h, [σ])[1][1], "Mooncake" => h -> mooncake_gradient(h, [σ])[1])[backend]
        @test gradient(score) ≈ exact rtol = 1e-10
        @test gradient(scores) ≈ exact_sum rtol = 1e-10
    end
    # FFTDensityInverter prices with the density it returns
    @test pricer(model(σ), ZeroRate(r), FFTDensityInverter(), EuropeanOption(T, 105.0)) ≈ pricer(model(σ), ZeroRate(r), CosDensityMethod(), EuropeanOption(T, 105.0)) atol = 1e-9
    @test_throws MethodError density(model(σ), ZeroRate(r), LewisMethod(600.0, 20000), T)
end
