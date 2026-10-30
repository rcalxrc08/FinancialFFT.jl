# Models without a closed formula: every method is compared with LewisMethod, which matches
# blsprice / blsbin to machine precision in Black Scholes. COS and Lewis agree to ~1e-13 on Kou, VG and NIG.
@testset "Other models vs LewisMethod" begin
    S0, r, d, T = 100.0, 0.02, 0.01, 1.0
    reference_method = LewisMethod(600.0, 20000)
    # (method, tolerance for EuropeanOption, tolerance for BinaryEuropeanOption); `nothing` when unsupported
    methods = [
        "CosMethod" => (CosMethod(2^12), 1e-5, 1e-5),
        "CarrMadanLewisMethod" => (CarrMadanLewisMethod(600.0, 16), 1e-7, Broken(1e-5)),
        "CarrMadanMethod" => (CarrMadanMethod(400.0, 18), 1e-5, nothing),
        "DensityInverter" => (FinancialFFT.DensityInverter(18, 3.0), 1e-7, 1e-4),
    ]
    for (model_name, model) in model_cases(S0, d), (method_name, (method, toll_eu, toll_bin)) in methods
        @testset "$model_name $method_name" begin
            for K in (90.0, 100.0, 110.0)
                for (opt, toll) in ((EuropeanOption(T, K), toll_eu), (EuropeanOption(T, K, false), toll_eu), (BinaryEuropeanOption(T, K), toll_bin), (BinaryEuropeanOption(T, K, false), toll_bin))
                    isnothing(toll) && continue
                    reference = pricer(model, ZeroRate(r), reference_method, opt)
                    # the left tail of Heston goes beyond the default truncation of the density
                    model_method = model isa HestonProcess && method isa FinancialFFT.DensityInverter ? FinancialFFT.DensityInverter(18, 5.0) : method
                    test_within(() -> abs(pricer(model, ZeroRate(r), model_method, opt) - reference), toll)
                end
            end
        end
    end
end

# Heston is not a Lévy process: its exponent is nonlinear in T. The reference is a direct integration of the Riccati equations.
@testset "Heston characteristic exponent" begin
    function riccati_exponent(z, σ, σ₀, λ, κ, ρ, θ, T; n = 5000)
        β = κ + λ - ρ * σ * z
        f(D) = σ^2 / 2 * D^2 - β * D + (z^2 - z) / 2
        C, D = zero(complex(z)), zero(complex(z))
        h = T / n
        for _ = 1:n
            k1 = f(D)
            D1 = D + h / 2 * k1
            k2 = f(D1)
            D2 = D + h / 2 * k2
            k3 = f(D2)
            D3 = D + h * k3
            k4 = f(D3)
            C += κ * θ * h / 6 * (D + 2 * D1 + 2 * D2 + D3)
            D += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        end
        return C + σ₀^2 * D
    end
    # (σ, σ₀, λ, κ, ρ, θ): the second one has κ + λ < ρσ, the third one κ + λ < 0
    @testset "Riccati equations, parameters $params" for params in ((0.5751, sqrt(0.0175), 0.0, 1.5768, -0.5711, 0.0398), (1.5, 0.3, 0.0, 0.5, 0.9, 0.04), (0.3, 0.2, -1.5, 1.0, -0.5, 0.04))
        model = HestonProcess(params..., Underlying(100.0, 0.0))
        for T in (0.1, 1.0, 5.0), z in (2.0im, -7.0im, 0.5 + 40.0im, 0.5 - 3.0im, 1.0 + 2.0im, 1.0 + 60.0im)
            @test abs(exp(FinancialFFT.characteristic_exponent_i(z, model, T)) - exp(riccati_exponent(z, params..., T))) < 1e-9
        end
    end
    model = HestonProcess(0.5751, sqrt(0.0175), 0.0, 1.5768, -0.5711, 0.0398, Underlying(100.0, 0.0))
    @testset "Real arguments" begin
        # γ is imaginary for the largest ones
        for z in (-3.0, -1.0, 0.5, 1.5, 3.0, 6.0)
            @test FinancialFFT.characteristic_exponent_i(z, model, 1.0) ≈ real(riccati_exponent(complex(z), 0.5751, sqrt(0.0175), 0.0, 1.5768, -0.5711, 0.0398, 1.0)) rtol = 1e-8
        end
        @test iszero(FinancialFFT.characteristic_exponent_i(1, model, 1.0))
        # γ = 0 in 1 when κ + λ = ρσ
        @test iszero(FinancialFFT.characteristic_exponent_i(1, HestonProcess(0.6, 0.2, 0.0, 0.3, 0.5, 0.04, Underlying(100.0, 0.0)), 1.0))
    end
    @testset "Vectorized function matches the scalar one" begin
        zs = collect(0.5 .+ range(0.0, 50.0, length = 11) .* im)
        @test FinancialFFT.characteristic_exponent_vi(zs, model, 1.3) ≈ [FinancialFFT.characteristic_exponent_i(z, model, 1.3) for z in zs]
    end
    # Reference value of the COS paper, Fang & Oosterlee (2008)
    @testset "Benchmark, $method_name" for (method_name, method) in (("LewisMethod", LewisMethod(600.0, 20000)), ("CosMethod", CosMethod(2^12)), ("CarrMadanLewisMethod", CarrMadanLewisMethod(600.0, 16)))
        @test abs(pricer(model, ZeroRate(0.0), method, EuropeanOption(1.0, 100.0)) - 5.7851554) < 1e-6
    end
end

# A truncation range on the wrong side of the heavy tail cuts real probability mass, which symmetric models do not show.
@testset "CosMethod on skewed models vs LewisMethod" begin
    S0, r, d = 100.0, 0.02, 0.01
    models = ("VarianceGamma" => VarianceGammaProcess(0.12, -0.3, 0.5, Underlying(S0, d)), "Kou" => KouProcess(0.2, 5.0, 0.1, 50.0, 3.0, Underlying(S0, d)), "Heston" => HestonProcess(1.0, 0.2, 0.0, 1.5, -0.9, 0.04, Underlying(S0, d)))
    @testset "$model_name, T = $T" for (model_name, model) in models, T in (1.0, 5.0)
        for K in (80.0, 100.0, 120.0)
            opt = EuropeanOption(T, K)
            @test abs(pricer(model, ZeroRate(r), CosMethod(2^12), opt) - pricer(model, ZeroRate(r), LewisMethod(600.0, 20000), opt)) < 1e-4
        end
    end
end

# The undamped transform of the time value has a kink at the forward, and decays like 1 / v²: it misses this strike by S0 / (π A).
@testset "CarrMadanMethod damping" begin
    S0, r, d, T = 100.0, 0.02, 0.01, 1.0
    reference_method = LewisMethod(600.0, 20000)
    opt = EuropeanOption(T, S0 * exp((r - d) * T))
    @testset "At the forward, $model_name" for (model_name, model) in model_cases(S0, d)
        @test abs(pricer(model, ZeroRate(r), CarrMadanMethod(400.0, 18), opt) - pricer(model, ZeroRate(r), reference_method, opt)) < 1e-5
    end
    @testset "Damping below the moment bound" begin
        # the default damping would need E[S^1.75], which is not finite
        kou = KouProcess(0.2, 1.0, 0.1, 1.1, 20.0, Underlying(S0, d))
        @test FinancialFFT.carr_madan_damping(kou, T, 0.75) ≈ 0.05
        @test FinancialFFT.carr_madan_damping(kou, T, 0.01) == 0.01
        @test abs(pricer(kou, ZeroRate(r), CarrMadanMethod(400.0, 18), opt) - pricer(kou, ZeroRate(r), reference_method, opt)) < 1e-5
        heston = HestonProcess(1.5, 0.3, 0.0, 0.5, 0.9, 0.04, Underlying(S0, d))
        @test FinancialFFT.carr_madan_damping(heston, 2.0, 0.75) < FinancialFFT.carr_madan_damping(heston, 0.5, 0.75)
    end
    @testset "User defined damping" begin
        model = last(first(model_cases(S0, d)))
        @test abs(pricer(model, ZeroRate(r), CarrMadanMethod(400.0, 18, 0.5), opt) - pricer(model, ZeroRate(r), reference_method, opt)) < 1e-5
    end
end

# Sensitivities with respect to the rate and to the model parameters, against LewisMethod.
@testset "Other models sensitivities vs LewisMethod" begin
    S0, d, T, K = 100.0, 0.01, 1.0, 100.0
    reference_method = LewisMethod(600.0, 20000)
    opt = EuropeanOption(T, K)
    @testset "HyperDualNumbers rate, VarianceGamma $method_name" for (method_name, method, toll) in (("CosMethod", CosMethod(2^12), 1e-5), ("CarrMadanLewisMethod", CarrMadanLewisMethod(600.0, 18), 1e-3))
        model = VarianceGammaProcess(0.2, 0.03, 0.01, Underlying(S0, d))
        zero_rate = ZeroRate(hyper(0.02, 1.0, 1.0, 0.0))
        test_within(() -> max_abs_diff(pricer(model, zero_rate, method, opt), pricer(model, zero_rate, reference_method, opt)), toll)
    end
    kou_price(method) = (σ, λ, p, λ₊, λ₋) -> pricer(KouProcess(σ, λ, p, λ₊, λ₋, Underlying(S0, d)), ZeroRate(0.02), method, opt)
    kou_params = [0.2, 5.0, 0.3, 30.0, 20.0]
    reference = Zygote.gradient(kou_price(reference_method), kou_params...)
    @testset "Zygote Kou parameters, CosMethod" begin
        test_within(() -> max_abs_diff(collect(Zygote.gradient(kou_price(CosMethod(2^12)), kou_params...)), collect(reference)), 1e-5)
    end
    @testset "ForwardDiff Kou parameters, $method_name" for (method_name, method, toll) in (("LewisMethod", reference_method, 1e-8), ("CosMethod", CosMethod(2^12), 1e-5), ("CarrMadanLewisMethod", CarrMadanLewisMethod(600.0, 16), 1e-5), ("DensityInverter", FinancialFFT.DensityInverter(18, 3.0), 1e-5))
        test_within(() -> max_abs_diff(ForwardDiff.gradient(x -> kou_price(method)(x...), kou_params), collect(reference)), toll)
    end
    @testset "Mooncake Kou parameters, $method_name" for (method_name, method, toll) in (("LewisMethod", reference_method, 1e-8), ("CosMethod", CosMethod(2^12), 1e-5), ("CarrMadanLewisMethod", CarrMadanLewisMethod(600.0, 16), 1e-5), ("DensityInverter", FinancialFFT.DensityInverter(18, 3.0), 1e-5))
        test_within(() -> max_abs_diff(mooncake_gradient(x -> kou_price(method)(x...), kou_params), collect(reference)), toll)
    end
    @testset "ReverseDiff Kou parameters, $method_name" for (method_name, method, toll) in (("LewisMethod", reference_method, 1e-8), ("CosMethod", CosMethod(2^12), 1e-5), ("CarrMadanLewisMethod", CarrMadanLewisMethod(600.0, 16), 1e-5), ("DensityInverter", FinancialFFT.DensityInverter(18, 3.0), 1e-5))
        test_within(() -> max_abs_diff(ReverseDiff.gradient(x -> kou_price(method)(x...), kou_params), collect(reference)), toll)
    end
    # a recorded tape replayed at new inputs recomputes the fft in its forward pass
    @testset "ReverseDiff replayed tape, DensityInverter" begin
        f = x -> kou_price(FinancialFFT.DensityInverter(12, 3.0))(x...)
        tape = ReverseDiff.GradientTape(f, kou_params)
        new_params = kou_params .* 1.1
        @test ReverseDiff.gradient!(tape, new_params) ≈ ReverseDiff.gradient(f, new_params)
    end
    heston_price(method) = x -> pricer(HestonProcess(x[1], x[2], x[3], x[4], x[5], x[6], Underlying(S0, d)), ZeroRate(0.02), method, opt)
    heston_params = [0.3, 0.2, 0.0, 1.5, -0.7, 0.04]
    heston_reference = ForwardDiff.gradient(heston_price(reference_method), heston_params)
    @testset "Zygote Heston parameters, CosMethod" begin
        test_within(() -> max_abs_diff(Zygote.gradient(heston_price(CosMethod(2^12)), heston_params)[1], heston_reference), 1e-5)
    end
    @testset "Heston parameters, $method_name" for (method_name, method, toll) in (("CosMethod", CosMethod(2^12), 1e-5), ("CarrMadanLewisMethod", CarrMadanLewisMethod(600.0, 16), 1e-5), ("DensityInverter", FinancialFFT.DensityInverter(16, 3.0), 1e-4))
        f = heston_price(method)
        @testset "ForwardDiff" begin
            test_within(() -> max_abs_diff(ForwardDiff.gradient(f, heston_params), heston_reference), toll)
        end
        @testset "Mooncake" begin
            test_within(() -> max_abs_diff(mooncake_gradient(f, heston_params), heston_reference), toll)
        end
        @testset "ReverseDiff" begin
            test_within(() -> max_abs_diff(ReverseDiff.gradient(f, heston_params), heston_reference), toll)
        end
    end
end
