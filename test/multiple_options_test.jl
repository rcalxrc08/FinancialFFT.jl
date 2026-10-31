# Pricing several options at once must give the same prices as pricing them one by one.
@testset "Multiple options" begin
    S0, r, d = 100.0, 0.02, 0.01
    model = MertonProcess(0.2, 5.0, 0.03, 0.02, Underlying(S0, d))
    zero_rate = ZeroRate(r)
    scalar_prices(method, opts) = [pricer(model, zero_rate, method, opt) for opt in opts]

    @testset "Options with different maturities, $name" for (name, method) in ("LewisMethod" => LewisMethod(600.0, 20000), "CosMethod" => CosMethod(2^12), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 16), "CarrMadanMethod" => CarrMadanMethod(400.0, 18), "DensityInverter" => FinancialFFT.DensityInverter(18, 3.0))
        opts = [EuropeanOption(1.0, 100.0), EuropeanOption(0.5, 70.0), BinaryEuropeanOption(1.0, 120.0, false), EuropeanOption(2.0, 100.0), BinaryEuropeanOption(0.5, 100.0)]
        @test max_abs_diff(pricer.(model, zero_rate, method, opts), scalar_prices(method, opts)) < 1e-12
    end

    Ks = [80.0, 90.0, 100.0, 110.0, 120.0]
    calls = [true, false, true, false, true]
    @testset "Smiles, $name" for (name, method) in ("LewisMethod" => LewisMethod(600.0, 20000), "CosMethod" => CosMethod(2^12), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 16), "CarrMadanMethod" => CarrMadanMethod(400.0, 18), "DensityInverter" => FinancialFFT.DensityInverter(18, 3.0))
        smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls)
        @test max_abs_diff(pricer(model, zero_rate, method, smile), scalar_prices(method, EuropeanOption.(1.0, Ks, calls))) < 1e-12
        binary_smile = FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0, Ks, calls)
        @test max_abs_diff(pricer(model, zero_rate, method, binary_smile), scalar_prices(method, BinaryEuropeanOption.(1.0, Ks, calls))) < 1e-12
    end

    # A payoff without a formula must not fall back on the vanilla one.
    @test_throws MethodError pricer(model, zero_rate, CarrMadanLewisMethod(600.0, 16), BarrierOptionUpOut(1.0, 100.0, 120.0))

    # Several strikes through the batched fft, against the sum of the scalar prices.
    @testset "Smile gradients, DensityInverter, $kind" for (kind, smile_type, option_type) in (("european", FinancialMonteCarlo.EuropeanOptionSmile, EuropeanOption), ("binary", FinancialMonteCarlo.BinaryEuropeanOptionSmile, BinaryEuropeanOption))
        method = FinancialFFT.DensityInverter(12, 3.0)
        build(x) = MertonProcess(x[1], 5.0, 0.03, 0.02, Underlying(x[2], d))
        smile_sum(x) = sum(pricer(build(x), zero_rate, method, smile_type(1.0, x[3] .* Ks, calls)))
        scalar_sum(x) = sum(pricer(build(x), zero_rate, method, option_type(1.0, x[3] * K, iscall)) for (K, iscall) in zip(Ks, calls))
        x = [0.2, S0, 1.0]
        reference = ForwardDiff.gradient(scalar_sum, x)
        @test max_abs_diff(ForwardDiff.gradient(smile_sum, x), reference) < 1e-11
        @test max_abs_diff(ReverseDiff.gradient(smile_sum, x), reference) < 1e-11
        @test max_abs_diff(Zygote.gradient(smile_sum, x)[1], reference) < 1e-11
        @test max_abs_diff(mooncake_gradient(smile_sum, x), reference) < 1e-11
    end

    # The strikes of a smile share the grid of the extreme ones, which is wider than the default of each strike.
    @testset "Smile of DensityInverter with the default grid" begin
        sigma = 0.2
        bs = BlackScholesProcess(sigma, Underlying(S0, d))
        prices = pricer(bs, zero_rate, FinancialFFT.DensityInverter(18), FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls))
        @test max_abs_diff(prices, blsprice.(S0, Ks, r, 1.0, sigma, d, calls)) < 1e-7
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
