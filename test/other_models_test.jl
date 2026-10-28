# Models without a closed formula: every method is compared with LewisMethod, which matches
# blsprice / blsbin to machine precision in Black Scholes. COS and Lewis agree to ~1e-13 on Kou, VG and NIG.
@testset "Other models vs LewisMethod" begin
    S0, r, d, T = 100.0, 0.02, 0.01, 1.0
    reference_method = LewisMethod(600.0, 20000)
    # (method, tolerance for EuropeanOption, tolerance for BinaryEuropeanOption); `nothing` when unsupported
    methods = [
        "CosMethod" => (CosMethod(2^12), 1e-5, 1e-5),
        "CarrMadanLewisMethod" => (CarrMadanLewisMethod(600.0, 16), 1e-7, Broken(1e-5)),
        "CarrMadanMethod" => (CarrMadanMethod(400.0, 18), Broken(1e-2), nothing),
        "DensityInverter" => (FinancialFFT.DensityInverter(18, 3.0), 1e-7, 1e-4),
    ]
    for (model_name, model) in model_cases(S0, d), (method_name, (method, toll_eu, toll_bin)) in methods
        @testset "$model_name $method_name" begin
            for K in (90.0, 100.0, 110.0)
                for (opt, toll) in ((EuropeanOption(T, K), toll_eu), (EuropeanOption(T, K, false), toll_eu), (BinaryEuropeanOption(T, K), toll_bin), (BinaryEuropeanOption(T, K, false), toll_bin))
                    isnothing(toll) && continue
                    reference = pricer(model, ZeroRate(r), reference_method, opt)
                    test_within(() -> abs(pricer(model, ZeroRate(r), method, opt) - reference), toll)
                end
            end
        end
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
end
