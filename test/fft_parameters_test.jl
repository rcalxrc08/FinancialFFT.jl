# CarrMadanMethod(ε) and CarrMadanLewisMethod(ε): A and Npow chosen at pricing time from a price tolerance (issue #41);
# LewisMethod(ε): A and N (#48).
@testset "FFT parameters from a tolerance" begin
    S0, r, d = 100.0, 0.02, 0.01
    und = Underlying(S0, d)
    zero_rate = ZeroRate(r)
    Ks = collect(range(50.0, 200.0, length = 31))
    iscall = [isodd(i) for i in eachindex(Ks)]
    constructors = ("CarrMadanMethod" => CarrMadanMethod, "CarrMadanLewisMethod" => CarrMadanLewisMethod, "LewisMethod" => LewisMethod)
    # a node budget too small for ε = 1e-10
    too_few_nodes = Dict("CarrMadanMethod" => (Npow_max = 8,), "CarrMadanLewisMethod" => (Npow_max = 8,), "LewisMethod" => (N_max = 101,))
    smiles(T) = (:vanilla => FinancialMonteCarlo.EuropeanOptionSmile(T, Ks, iscall), :binary => FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, Ks, iscall))
    # the method with the grid that the tolerance gives for this model and payoff
    resolved(method::CarrMadanMethod, model, zr, opt; und) = CarrMadanMethod(FinancialFFT.fft_grid(method.grid, method, model, und, zr, opt)..., method.α)
    resolved(method::CarrMadanLewisMethod, model, zr, opt; und) = CarrMadanLewisMethod(FinancialFFT.fft_grid(method.grid, method, model, und, zr, opt)...)
    resolved(method::LewisMethod, model, zr, opt; und) = LewisMethod(FinancialFFT.quadrature_grid(method.grid, method, model, und, zr, opt)...)
    @testset "Black Scholes, $name, $payoff, T = $T, ε = $ε" for (name, ctor) in constructors, T in (0.1, 2.0), ε in (1e-5, 1e-9), (payoff, opt) in smiles(T)
        sigma = 0.2
        model = BlackScholesProcess(sigma)
        exact = payoff == :vanilla ? blsprice.(S0, Ks, r, T, sigma, d, iscall) : blsbin.(S0, Ks, r, T, sigma, d, iscall)
        @test maximum(abs.(pricer(model, und, zero_rate, ctor(ε), opt) .- exact)) < ε
    end
    reference = LewisMethod(2000.0, 200000)
    @testset "$model_name, $name, $payoff" for (model_name, model) in model_cases(), (name, ctor) in constructors, (payoff, opt) in smiles(1.0)
        ε = 1e-7
        @test maximum(abs.(pricer(model, und, zero_rate, ctor(ε), opt) .- pricer(model, und, zero_rate, reference, opt))) < ε
    end
    @testset "a single option, $name" for (name, ctor) in constructors
        model = BlackScholesProcess(0.2)
        opt = EuropeanOption(1.0, 120.0)
        @test abs(pricer(model, und, zero_rate, ctor(1e-9), opt) - blsprice(S0, 120.0, r, 1.0, 0.2, d)) < 1e-9
    end
    @testset "too many nodes, $name" for (name, ctor) in constructors
        model = BlackScholesProcess(0.2)
        @test_throws ArgumentError pricer(model, und, zero_rate, ctor(1e-10; too_few_nodes[name]...), EuropeanOption(1.0, 100.0))
    end
    # The parameters are constants: the gradients are those of the method with the same parameters, given, on every
    # back-end.
    @testset "AD, $name" for (name, ctor) in constructors
        opt = EuropeanOption(1.0, 100.0)
        x = [S0, r, 0.2]
        method = ctor(1e-9)
        price(method) = x -> pricer(BlackScholesProcess(x[3]), Underlying(x[1], d), ZeroRate(x[2]), method, opt)
        fixed = resolved(method, BlackScholesProcess(0.2), zero_rate, opt; und = Underlying(S0, d))
        @test fixed.grid.A isa Float64
        reference = ForwardDiff.gradient(price(fixed), x)
        @test abs(reference[3] - blsvega(S0, 100.0, r, 1.0, 0.2, d)) < 1e-7
        @test maximum(abs.(ForwardDiff.gradient(price(method), x) .- reference)) < 1e-12
        @test maximum(abs.(ReverseDiff.gradient(price(method), x) .- reference)) < 1e-12
        @test maximum(abs.(only(Zygote.gradient(price(method), x)) .- reference)) < 1e-12
        @test maximum(abs.(mooncake_gradient(price(method), x) .- reference)) < 1e-12
    end
    @testset "Float32, $name, $payoff" for (name, ctor) in constructors, payoff in (:vanilla, :binary)
        model = BlackScholesProcess(0.2f0)
        zr = ZeroRate(0.02f0)
        opt = payoff == :vanilla ? FinancialMonteCarlo.EuropeanOptionSmile(1.0f0, Float32.(Ks), iscall) : FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0f0, Float32.(Ks), iscall)
        method = ctor(1e-3)
        @test resolved(method, model, zr, opt; und = Underlying(100.0f0, 0.01f0)).grid.A isa Float32
        prices = pricer(model, Underlying(100.0f0, 0.01f0), zr, method, opt)
        @test eltype(prices) == Float32
        exact = payoff == :vanilla ? blsprice.(S0, Ks, r, 1.0, 0.2, d, iscall) : blsbin.(S0, Ks, r, 1.0, 0.2, d, iscall)
        @test maximum(abs.(prices .- exact)) < 1e-3
    end
    # LewisMethod(ε) is not wasteful: on Black Scholes its error is above ε / 10 (0.4-0.6 ε measured, 2026-11-26), and a
    # given grid stays the given one.
    @testset "LewisMethod(ε), error near ε, T = $T, $payoff" for T in (0.1, 2.0), (payoff, opt) in smiles(T)
        sigma, ε = 0.2, 1e-9
        exact = payoff == :vanilla ? blsprice.(S0, Ks, r, T, sigma, d, iscall) : blsbin.(S0, Ks, r, T, sigma, d, iscall)
        error = maximum(abs.(pricer(BlackScholesProcess(sigma), und, zero_rate, LewisMethod(ε), opt) .- exact))
        @test ε / 10 < error < ε
    end
    @test FinancialFFT.quadrature_grid(LewisMethod(600.0, 20000).grid, LewisMethod(600.0, 20000), BlackScholesProcess(0.2), und, zero_rate, EuropeanOption(1.0, 100.0)) == (600.0, 20000)
    @test LewisMethod(1e-8) isa LewisMethod{<:FinancialFFT.QuadratureTolerance}
    @test (@inferred pricer(BlackScholesProcess(0.2), und, zero_rate, LewisMethod(1e-8), last(smiles(1.0)[1]))) isa Vector{Float64}
    @test_throws DomainError LewisMethod(-1e-8)
    # A low-variance Heston (4% volatility, T = 1/6): its exponent decays slowly along the line 1/2 + iv, so the tolerance
    # asks for A > 2000, where LewisMethod(100, 4001) was 18% off at the money (#48, FinancialCalibrator tests).
    @testset "low-variance Heston" begin
        model = HestonProcess(0.3, 0.04, 1.0, 0.04, -0.5, 0.04)
        und0, zr0 = Underlying(100.0, 0.0), ZeroRate(0.0)
        opt = FinancialMonteCarlo.EuropeanOptionSmile(1 / 6, [80.0, 100.0], [false, false])
        method = LewisMethod(1e-8)
        @test first(FinancialFFT.quadrature_grid(method.grid, method, model, und0, zr0, opt)) > 2000
        @test maximum(abs.(pricer(model, und0, zr0, method, opt) .- pricer(model, und0, zr0, CosMethod(2^14), opt))) < 1e-8
    end
    # The aliasing bound rests on this: the trapezoidal weights (#104) add copies of the damped call at the multiples of 2b
    # (b = π 2^Npow / A, the half-width of the strike grid), with weight 1, and none at the odd multiples of b, which
    # Simpson's weights added with weight -1/3. A small b makes them dominate the error, which they predict up to the
    # interpolation error.
    @testset "trapezoidal copies at the period" begin
        # b = 4.02; A = 200 keeps the interpolation error of the copies (which vary fast in k) at 4e-7 (3e-5 at A = 100)
        sigma, T, α, A, Npow = 0.2, 1.0, 0.75, 200.0, 8
        model = BlackScholesProcess(sigma)
        b = π * 2^Npow / A
        call(k) = blsprice(S0, S0 * exp(k), r, T, sigma, d)
        copies(k) = sum(exp(α * 2b * m) * call(k + 2b * m) for m = -6:6 if m != 0)
        ks = collect(range(-0.6b, 0.6b, length = 13))
        opt = FinancialMonteCarlo.EuropeanOptionSmile(T, S0 .* exp.(ks), fill(true, length(ks)))
        error = pricer(model, und, zero_rate, CarrMadanMethod(A, Npow, α), opt) .- call.(ks)
        @test maximum(abs.(error)) > 1e-2
        @test maximum(abs.(error .- copies.(ks))) < 1e-5
    end
end
