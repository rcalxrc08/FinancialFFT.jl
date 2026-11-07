# Prices keep the floating-point type of the inputs: no Float64 constant of a method promotes Float32 or BigFloat.
# The FFT methods transform BigFloat with FFTA. In BigFloat the truncation range of the COS methods
# is wider (its tail probability is scaled by the machine epsilon), so CosDensityMethod has a larger Simpson error.
@testset "Floating-point types" begin
    type_methods = ["LewisMethod" => LewisMethod(600.0, 2000), "CosMethod" => CosMethod(2^8), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 12), "CarrMadanMethod" => CarrMadanMethod(400.0, 12), "FFTDensityInverter" => FFTDensityInverter(12, 2^8), "CosDensityMethod" => CosDensityMethod(64, 2^8)]
    type_payoffs(Tn) = [EuropeanOption(Tn(1), Tn(100)), BinaryEuropeanOption(Tn(1), Tn(95), false), FinancialMonteCarlo.EuropeanOptionSmile(Tn(1), Tn[90, 110], [true, false]), FinancialMonteCarlo.BinaryEuropeanOptionSmile(Tn(1), Tn[90, 110], [true, false])]
    type_model(Tn) = MertonProcess(Tn(0.2), Tn(5), Tn(0.03), Tn(0.02), Underlying(Tn(100), Tn(0.01)))
    setprecision(BigFloat, 128) do
        @testset "$name" for (name, method) in type_methods
            for (opt64, opt32, optbig) in zip(type_payoffs(Float64), type_payoffs(Float32), type_payoffs(BigFloat))
                price64 = pricer(type_model(Float64), ZeroRate(0.02), method, opt64)
                price32 = pricer(type_model(Float32), ZeroRate(0.02f0), method, opt32)
                pricebig = pricer(type_model(BigFloat), ZeroRate(BigFloat(0.02)), method, optbig)
                @test eltype(price32) == Float32
                @test eltype(pricebig) == BigFloat
                @test max_abs_diff(price32, price64) < 1e-3
                @test max_abs_diff(pricebig, price64) < (name in ("CosDensityMethod", "FFTDensityInverter") ? 1e-6 : 1e-11)
            end
        end
        # BigFloat derivatives through the FFTs of FFTA, out of place (the extensions of the forward back-ends wrap its plan
        # for their fft!), against Float64: the delta.
        @testset "$name, BigFloat derivatives, $backend" for (name, method) in type_methods[3:5], backend in ("ForwardDiff", "DualNumbers", "HyperDualNumbers", "TaylorSeries", "Zygote")
            price(s, Tn) = pricer(MertonProcess(Tn(0.2), Tn(5), Tn(0.03), Tn(0.02), Underlying(s, Tn(0.01))), ZeroRate(Tn(0.02)), method, EuropeanOption(Tn(1), Tn(100)))
            derivative = Dict("ForwardDiff" => (f, s) -> ForwardDiff.derivative(f, s), "DualNumbers" => (f, s) -> DualNumbers.epsilon(f(Dual(s, one(s)))), "HyperDualNumbers" => (f, s) -> HyperDualNumbers.eps1(f(hyper(s, one(s), one(s), zero(s)))), "TaylorSeries" => (f, s) -> TaylorSeries.getcoeff(f(s + Taylor1(typeof(s), 1)), 1), "Zygote" => (f, s) -> Zygote.gradient(f, s)[1])[backend]
            deltabig = derivative(s -> price(s, BigFloat), BigFloat(100))
            @test deltabig isa BigFloat
            @test abs(deltabig - derivative(s -> price(s, Float64), 100.0)) < 1e-9
        end
    end
    # Type stability: the prices and the densities are inferred (a variable assigned both inside a closure of
    # @ignore_derivatives and outside of it is boxed, and hides every type after it).
    @testset "Inferred, $(case.name)" for case in METHOD_CASES
        model, zero_rate = type_model(Float64), ZeroRate(0.02)
        for opt in type_payoffs(Float64)
            @test (@inferred pricer(model, zero_rate, case.method, opt)) isa Union{Float64, Vector{Float64}}
        end
        if case.method isa FinancialFFT.AbstractDensityMethod
            f = @inferred density(model, zero_rate, case.method, 1.0)
            @test (@inferred FinancialFFT.density_values(f, [-0.1, 0.0, 0.1])) isa Vector{Float64}
        end
    end
end
