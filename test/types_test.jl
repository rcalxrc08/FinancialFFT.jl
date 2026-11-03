# Prices keep the floating-point type of the inputs: no Float64 constant of a method promotes Float32 or BigFloat.
# The FFT methods transform BigFloat with the generic radix-2 FFT. In BigFloat the truncation range of the COS methods
# is wider (its tail probability is scaled by the machine epsilon), so CosDensityMethod has a larger Simpson error.
@testset "Floating-point types" begin
    type_methods = ["LewisMethod" => LewisMethod(600.0, 2000), "CosMethod" => CosMethod(2^8), "CarrMadanLewisMethod" => CarrMadanLewisMethod(600.0, 12), "CarrMadanMethod" => CarrMadanMethod(400.0, 12), "DensityInverter" => FinancialFFT.DensityInverter(12, 3.0), "CosDensityMethod" => CosDensityMethod(64, 2^8)]
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
                @test max_abs_diff(pricebig, price64) < (name == "CosDensityMethod" ? 1e-6 : 1e-11)
            end
        end
    end
end
