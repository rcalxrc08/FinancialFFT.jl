# Pricing several options at once must give the same prices as pricing them one by one.
@testset "Multiple options" begin
    S0, r, d = 100.0, 0.02, 0.01
    model = MertonProcess(0.2, 5.0, 0.03, 0.02, Underlying(S0, d))
    zero_rate = ZeroRate(r)
    scalar_prices(method, opts) = [pricer(model, zero_rate, method, opt) for opt in opts]

    @testset "Options with different maturities, $name" for (name, method) in ("LewisMethod" => LewisMethod(600.0, 20000), "CosMethod" => CosMethod(2^12), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 16), "CarrMadanMethod" => CarrMadanMethod(400.0, 18), "FFTDensityInverter" => FFTDensityInverter(), "CosDensityMethod" => CosDensityMethod())
        opts = [EuropeanOption(1.0, 100.0), EuropeanOption(0.5, 70.0), BinaryEuropeanOption(1.0, 120.0, false), EuropeanOption(2.0, 100.0), BinaryEuropeanOption(0.5, 100.0)]
        @test max_abs_diff(pricer.(model, zero_rate, method, opts), scalar_prices(method, opts)) < 1e-12
    end

    Ks = [80.0, 90.0, 100.0, 110.0, 120.0]
    calls = [true, false, true, false, true]
    @testset "Smiles, $name" for (name, method) in ("LewisMethod" => LewisMethod(600.0, 20000), "CosMethod" => CosMethod(2^12), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 16), "CarrMadanMethod" => CarrMadanMethod(400.0, 18), "FFTDensityInverter" => FFTDensityInverter(), "CosDensityMethod" => CosDensityMethod())
        smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls)
        @test max_abs_diff(pricer(model, zero_rate, method, smile), scalar_prices(method, EuropeanOption.(1.0, Ks, calls))) < 1e-12
        binary_smile = FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0, Ks, calls)
        @test max_abs_diff(pricer(model, zero_rate, method, binary_smile), scalar_prices(method, BinaryEuropeanOption.(1.0, Ks, calls))) < 1e-12
    end

    # Outside SerialMode the smiles of LewisMethod and CosMethod are one broadcast over terms and strikes (the path of the
    # GPU, #46) instead of the Clenshaw recurrence: MultiThreading takes it on the CPU.
    @testset "Smiles outside SerialMode, $name, N = $N" for (name, method_N) in ("LewisMethod" => N -> LewisMethod(600.0, N), "CosMethod" => N -> CosMethod(N)), N in (4096, 4097)
        method = method_N(N)
        for smile in (FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls), FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0, Ks, calls))
            @test max_abs_diff(pricer(model, zero_rate, method, smile, FinancialMonteCarlo.MultiThreading()), pricer(model, zero_rate, method, smile)) < 1e-12
        end
    end

    # A payoff without a formula must not fall back on the vanilla one.
    @test_throws MethodError pricer(model, zero_rate, CarrMadanLewisMethod(600.0, 16), BarrierOptionUpOut(1.0, 100.0, 120.0))

    # Several strikes on one density, against the sum of the scalar prices.
    @testset "Smile gradients, FFTDensityInverter, $kind" for (kind, smile_type, option_type) in (("european", FinancialMonteCarlo.EuropeanOptionSmile, EuropeanOption), ("binary", FinancialMonteCarlo.BinaryEuropeanOptionSmile, BinaryEuropeanOption))
        method = FFTDensityInverter()
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

    # The strikes of a smile share the density, whose range does not depend on them.
    @testset "Smile of FFTDensityInverter in Black Scholes" begin
        sigma = 0.2
        bs = BlackScholesProcess(sigma, Underlying(S0, d))
        prices = pricer(bs, zero_rate, FFTDensityInverter(), FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls))
        @test max_abs_diff(prices, blsprice.(S0, Ks, r, 1.0, sigma, d, calls)) < 1e-8
    end

    # The truncation range shrinks with the maturity and, for Kou, is limited by the exponential moment bound.
    @testset "Short maturities, CosMethod, $model_name" for (model_name, short_model) in model_cases(S0, d)
        for T in (0.05, 0.25, 0.5)
            opt = EuropeanOption(T, 100.0)
            @test abs(pricer(short_model, zero_rate, CosMethod(2^12), opt) - pricer(short_model, zero_rate, LewisMethod(600.0, 20000), opt)) < 1e-5
        end
    end
    # The truncation range covers the forward of every strike, however far it is from S0: at short maturities these
    # strikes are many half widths away from it.
    @testset "Strike far from the spot, CosMethod" begin
        @testset "K = $K, isCall = $isCall" for K in (50.0, 80.0, 120.0, 300.0), isCall in (true, false)
            opt = EuropeanOption(0.01, K, isCall)
            @test abs(pricer(model, zero_rate, CosMethod(2^12), opt) - pricer(model, zero_rate, LewisMethod(600.0, 20000), opt)) < 1e-5
        end
        # one range for the whole smile, wider than the range of each of its strikes
        Ks = [50.0, 80.0, 100.0, 120.0, 300.0]
        isCall = [true, false, true, false, true]
        for (smile, payoff) in ((FinancialMonteCarlo.EuropeanOptionSmile(0.01, Ks, isCall), EuropeanOption), (FinancialMonteCarlo.BinaryEuropeanOptionSmile(0.01, Ks, isCall), BinaryEuropeanOption))
            @test maximum(abs.(pricer(model, zero_rate, CosMethod(2^12), smile) .- pricer.(model, zero_rate, CosMethod(2^12), payoff.(0.01, Ks, isCall)))) < 1e-10
        end
    end
end
