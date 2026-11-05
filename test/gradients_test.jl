# Gradients with respect to (S0, K, r, T, σ, d) against the gradient of blsprice / blsbin.
@testset "Black Scholes gradients vs FinancialToolbox" begin
    args = bs_args()
    for case in METHOD_CASES, (kind, toll) in case.payoffs, iscall in (true, false)
        f = fft_bs_pricer(case.method, kind, iscall)
        g = analytic_bs_pricer(kind, iscall)
        reference = ForwardDiff.gradient(x -> g(x...), args)
        @testset "$(case.name) $kind $(iscall ? "call" : "put")" begin
            @testset "ForwardDiff" begin
                test_within(() -> max_abs_diff(ForwardDiff.gradient(x -> f(x...), args), reference), toll.forwarddiff)
            end
            # Nested duals, which reach the fft of the FFT methods as arrays of duals. The puts are parity.
            iscall && @testset "ForwardDiff Hessian" begin
                test_within(() -> max_abs_diff(ForwardDiff.hessian(x -> f(x...), args), ForwardDiff.hessian(x -> g(x...), args)), toll.hessian)
            end
            @testset "ReverseDiff" begin
                test_within(() -> max_abs_diff(ReverseDiff.gradient(x -> f(x...), args), reference), toll.reversediff)
            end
            @testset "Zygote" begin
                test_within(() -> max_abs_diff([something(x, 0.0) for x in Zygote.gradient(f, args...)], reference), toll.zygote)
            end
            @testset "Mooncake" begin
                test_within(() -> max_abs_diff(mooncake_gradient(x -> f(x...), args), reference), toll.mooncake)
            end
        end
    end
end

# Zygote through the Clenshaw recurrences of the smiles (SerialMode) goes through their reverse rules (#47): against
# ForwardDiff, on smiles of several strikes, vanilla and binary, odd and even N.
@testset "Zygote through the Clenshaw recurrences, $name" for (name, method) in ("LewisMethod" => LewisMethod(600.0, 20000), "LewisMethod, odd N" => LewisMethod(600.0, 20001), "CosMethod" => CosMethod(2^12), "CosDensityMethod" => CosDensityMethod())
    Ks = collect(range(70.0, 140.0, length = 8))
    iscall = [isodd(i) for i in eachindex(Ks)]
    params = [100.0, 0.02, 0.2, 5.0, 0.03, 0.02]
    merton(x) = MertonProcess(x[3], x[4], x[5], x[6], Underlying(x[1], 0.01))
    for smile in (FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, iscall), FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0, Ks, iscall))
        price = x -> sum(pricer(merton(x), ZeroRate(x[2]), method, smile))
        @test max_abs_diff(only(Zygote.gradient(price, params)), ForwardDiff.gradient(price, params)) < 1e-9
    end
end
