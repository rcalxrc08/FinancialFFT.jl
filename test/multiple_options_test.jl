# Pricing several options at once must give the same prices as pricing them one by one.
@testset "Multiple options" begin
    S0, r, d = 100.0, 0.02, 0.01
    model = MertonProcess(0.2, 5.0, 0.03, 0.02, Underlying(S0, d))
    zero_rate = ZeroRate(r)
    scalar_prices(method, opts) = [pricer(model, zero_rate, method, opt) for opt in opts]

    @testset "Array of options with different maturities, $name" for (name, method) in ("LewisMethod" => LewisMethod(600.0, 20000), "CosMethod" => CosMethod(2^12), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 16), "CarrMadanMethod" => CarrMadanMethod(400.0, 18))
        opts = [EuropeanOption(1.0, 100.0), EuropeanOption(0.5, 70.0), EuropeanOption(1.0, 120.0, false), EuropeanOption(2.0, 100.0), EuropeanOption(0.5, 100.0)]
        @test max_abs_diff(pricer(model, zero_rate, method, opts), scalar_prices(method, opts)) < 1e-12
    end

    Ks = [80.0, 90.0, 100.0, 110.0, 120.0]
    calls = [true, false, true, false, true]
    @testset "Smiles, $name" for (name, method) in ("LewisMethod" => LewisMethod(600.0, 20000), "CosMethod" => CosMethod(2^12), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 16))
        smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls)
        @test max_abs_diff(pricer(model, zero_rate, method, smile), scalar_prices(method, EuropeanOption.(1.0, Ks, calls))) < 1e-12
        binary_smile = FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0, Ks, calls)
        @test max_abs_diff(pricer(model, zero_rate, method, binary_smile), scalar_prices(method, BinaryEuropeanOption.(1.0, Ks, calls))) < 1e-12
    end

    @testset "Vector of strikes, $name" for (name, method) in ("CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 16), "CarrMadanMethod" => CarrMadanMethod(400.0, 18))
        @test max_abs_diff(pricer(model, Ks, zero_rate, 1.0, method), scalar_prices(method, EuropeanOption.(1.0, Ks))) < 1e-12
    end

    # The truncation range shrinks with the maturity and, for Kou, is limited by the exponential moment bound.
    @testset "Short maturities, CosMethod, $model_name" for (model_name, short_model) in model_cases(S0, d)
        for T in (0.05, 0.25, 0.5)
            opt = EuropeanOption(T, 100.0)
            @test abs(pricer(short_model, zero_rate, CosMethod(2^12), opt) - pricer(short_model, zero_rate, LewisMethod(600.0, 20000), opt)) < 1e-5
        end
    end
    # The truncation range is centered on the forward and shared by all the strikes: a strike with |log(S0 / K)| larger
    # than the half width of the range is outside of it, and the cosine expansion aliases.
    @testset "Strike outside of the truncation range, CosMethod" begin
        opt = EuropeanOption(0.01, 80.0)
        test_within(() -> abs(pricer(model, zero_rate, CosMethod(2^12), opt) - pricer(model, zero_rate, LewisMethod(600.0, 20000), opt)), Broken(1e-5))
    end
end
