# A and Npow of the Carr-Madan methods from a price tolerance (issue #41).
@testset "FFT parameters from a tolerance" begin
    S0, r, d = 100.0, 0.02, 0.01
    zero_rate = ZeroRate(r)
    Ks = collect(range(50.0, 200.0, length = 31))
    iscall = [isodd(i) for i in eachindex(Ks)]
    constructors = ("CarrMadanMethod" => CarrMadanMethod, "CarrMadanLewisMethod" => CarrMadanLewisMethod)
    smiles(T) = (:vanilla => FinancialMonteCarlo.EuropeanOptionSmile(T, Ks, iscall), :binary => FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, Ks, iscall))
    @testset "Black Scholes, $name, $payoff, T = $T, ε = $ε" for (name, ctor) in constructors, T in (0.1, 2.0), ε in (1e-5, 1e-9), (payoff, opt) in smiles(T)
        sigma = 0.2
        model = BlackScholesProcess(sigma, Underlying(S0, d))
        exact = payoff == :vanilla ? blsprice.(S0, Ks, r, T, sigma, d, iscall) : blsbin.(S0, Ks, r, T, sigma, d, iscall)
        method = ctor(model, zero_rate, opt, ε)
        @test maximum(abs.(pricer(model, zero_rate, method, opt) .- exact)) < ε
    end
    reference = LewisMethod(2000.0, 200000)
    @testset "$model_name, $name, $payoff" for (model_name, model) in model_cases(S0, d), (name, ctor) in constructors, (payoff, opt) in smiles(1.0)
        ε = 1e-7
        method = ctor(model, zero_rate, opt, ε)
        @test maximum(abs.(pricer(model, zero_rate, method, opt) .- pricer(model, zero_rate, reference, opt))) < ε
    end
    @testset "a single option, $name" for (name, ctor) in constructors
        model = BlackScholesProcess(0.2, Underlying(S0, d))
        opt = EuropeanOption(1.0, 120.0)
        method = ctor(model, zero_rate, opt, 1e-9)
        @test abs(pricer(model, zero_rate, method, opt) - blsprice(S0, 120.0, r, 1.0, 0.2, d)) < 1e-9
    end
    @testset "too many nodes, $name" for (name, ctor) in constructors
        model = BlackScholesProcess(0.2, Underlying(S0, d))
        @test_throws ArgumentError ctor(model, zero_rate, EuropeanOption(1.0, 100.0), 1e-10; Npow_max = 8)
    end
    # the parameters are constants: built inside a differentiated function, they do not change the gradient
    @testset "ForwardDiff through the constructor, $name" for (name, ctor) in constructors
        opt = EuropeanOption(1.0, 100.0)
        price(sigma) = (model = BlackScholesProcess(sigma, Underlying(S0, d)); pricer(model, zero_rate, ctor(model, zero_rate, opt, 1e-9), opt))
        method = ctor(BlackScholesProcess(0.2, Underlying(S0, d)), zero_rate, opt, 1e-9)
        @test method.A isa Float64
        @test abs(ForwardDiff.derivative(price, 0.2) - blsvega(S0, 100.0, r, 1.0, 0.2, d)) < 1e-7
    end
end
