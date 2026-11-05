# CarrMadanMethod(ε) and CarrMadanLewisMethod(ε): A and Npow chosen at pricing time from a price tolerance (issue #41).
@testset "FFT parameters from a tolerance" begin
    S0, r, d = 100.0, 0.02, 0.01
    zero_rate = ZeroRate(r)
    Ks = collect(range(50.0, 200.0, length = 31))
    iscall = [isodd(i) for i in eachindex(Ks)]
    constructors = ("CarrMadanMethod" => CarrMadanMethod, "CarrMadanLewisMethod" => CarrMadanLewisMethod)
    smiles(T) = (:vanilla => FinancialMonteCarlo.EuropeanOptionSmile(T, Ks, iscall), :binary => FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, Ks, iscall))
    # the method with the grid that the tolerance gives for this model and payoff
    resolved(method::CarrMadanMethod, model, zr, opt) = CarrMadanMethod(FinancialFFT.fft_grid(method.grid, method, model, zr, opt)..., method.α)
    resolved(method::CarrMadanLewisMethod, model, zr, opt) = CarrMadanLewisMethod(FinancialFFT.fft_grid(method.grid, method, model, zr, opt)...)
    @testset "Black Scholes, $name, $payoff, T = $T, ε = $ε" for (name, ctor) in constructors, T in (0.1, 2.0), ε in (1e-5, 1e-9), (payoff, opt) in smiles(T)
        sigma = 0.2
        model = BlackScholesProcess(sigma, Underlying(S0, d))
        exact = payoff == :vanilla ? blsprice.(S0, Ks, r, T, sigma, d, iscall) : blsbin.(S0, Ks, r, T, sigma, d, iscall)
        @test maximum(abs.(pricer(model, zero_rate, ctor(ε), opt) .- exact)) < ε
    end
    reference = LewisMethod(2000.0, 200000)
    @testset "$model_name, $name, $payoff" for (model_name, model) in model_cases(S0, d), (name, ctor) in constructors, (payoff, opt) in smiles(1.0)
        ε = 1e-7
        @test maximum(abs.(pricer(model, zero_rate, ctor(ε), opt) .- pricer(model, zero_rate, reference, opt))) < ε
    end
    @testset "a single option, $name" for (name, ctor) in constructors
        model = BlackScholesProcess(0.2, Underlying(S0, d))
        opt = EuropeanOption(1.0, 120.0)
        @test abs(pricer(model, zero_rate, ctor(1e-9), opt) - blsprice(S0, 120.0, r, 1.0, 0.2, d)) < 1e-9
    end
    @testset "too many nodes, $name" for (name, ctor) in constructors
        model = BlackScholesProcess(0.2, Underlying(S0, d))
        @test_throws ArgumentError pricer(model, zero_rate, ctor(1e-10; Npow_max = 8), EuropeanOption(1.0, 100.0))
    end
    # The parameters are constants: the gradients are those of the method with the same parameters, given, on every
    # back-end.
    @testset "AD, $name" for (name, ctor) in constructors
        opt = EuropeanOption(1.0, 100.0)
        x = [S0, r, 0.2]
        method = ctor(1e-9)
        price(method) = x -> pricer(BlackScholesProcess(x[3], Underlying(x[1], d)), ZeroRate(x[2]), method, opt)
        fixed = resolved(method, BlackScholesProcess(0.2, Underlying(S0, d)), zero_rate, opt)
        @test fixed.grid.A isa Float64
        reference = ForwardDiff.gradient(price(fixed), x)
        @test abs(reference[3] - blsvega(S0, 100.0, r, 1.0, 0.2, d)) < 1e-7
        @test maximum(abs.(ForwardDiff.gradient(price(method), x) .- reference)) < 1e-12
        @test maximum(abs.(ReverseDiff.gradient(price(method), x) .- reference)) < 1e-12
        @test maximum(abs.(only(Zygote.gradient(price(method), x)) .- reference)) < 1e-12
        @test maximum(abs.(mooncake_gradient(price(method), x) .- reference)) < 1e-12
    end
    @testset "Float32, $name, $payoff" for (name, ctor) in constructors, payoff in (:vanilla, :binary)
        model = BlackScholesProcess(0.2f0, Underlying(100.0f0, 0.01f0))
        zr = ZeroRate(0.02f0)
        opt = payoff == :vanilla ? FinancialMonteCarlo.EuropeanOptionSmile(1.0f0, Float32.(Ks), iscall) : FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0f0, Float32.(Ks), iscall)
        method = ctor(1e-3)
        @test resolved(method, model, zr, opt).grid.A isa Float32
        prices = pricer(model, zr, method, opt)
        @test eltype(prices) == Float32
        exact = payoff == :vanilla ? blsprice.(S0, Ks, r, 1.0, 0.2, d, iscall) : blsbin.(S0, Ks, r, 1.0, 0.2, d, iscall)
        @test maximum(abs.(prices .- exact)) < 1e-3
    end
    # The aliasing bound rests on this: Simpson's weights are the trapezoidal ones times 1 - (-1)^j / 3, so besides the
    # copies of the damped call at the multiples of 2b (b = π 2^Npow / A, the half-width of the strike grid) there are
    # copies at the odd multiples of b with weight -1/3. A small b makes them dominate the error, which they predict up to
    # the interpolation error.
    @testset "Simpson copies at half the period" begin
        sigma, T, α, A, Npow = 0.2, 1.0, 0.75, 100.0, 8
        model = BlackScholesProcess(sigma, Underlying(S0, d))
        b = π * 2^Npow / A
        call(k) = blsprice(S0, S0 * exp(k), r, T, sigma, d)
        copies(k) = sum(exp(α * 2b * m) * call(k + 2b * m) for m = -6:6 if m != 0) - sum(exp(α * (2m - 1) * b) * call(k + (2m - 1) * b) for m = -6:6) / 3
        ks = collect(range(-0.6b, 0.6b, length = 13))
        opt = FinancialMonteCarlo.EuropeanOptionSmile(T, S0 .* exp.(ks), fill(true, length(ks)))
        error = pricer(model, zero_rate, CarrMadanMethod(A, Npow, α), opt) .- call.(ks)
        @test maximum(abs.(error)) > 1e-2
        @test maximum(abs.(error .- copies.(ks))) < 1e-5
    end
end
