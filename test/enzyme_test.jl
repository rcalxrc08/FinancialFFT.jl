# Enzyme (#62), through the EnzymeCore extension: gradients with respect to (S0, K, r, T, σ, d), in forward and reverse
# mode, against the gradient of blsprice / blsbin. Enzyme is loaded by this suite only.
using Enzyme

# The pricer of fft_bs_pricer, type stable for Enzyme, which fails on the Union of payoff types that build_payoff returns
# for a kind known at run time only, and on the splatting call f(x...): the kind is a type, the arguments indexed.
enzyme_payoff(::Val{:european}, T, K, iscall) = EuropeanOption(T, K, iscall)
enzyme_payoff(::Val{:binary}, T, K, iscall) = BinaryEuropeanOption(T, K, iscall)
enzyme_payoff(::Val{:european_smile}, T, K, iscall) = FinancialMonteCarlo.EuropeanOptionSmile(T, [K], [iscall])
enzyme_payoff(::Val{:binary_smile}, T, K, iscall) = FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, [K], [iscall])
enzyme_bs_pricer(method, kind::Val, iscall) = x -> first(pricer(BlackScholesProcess(x[5], Underlying(x[1], x[6])), ZeroRate(x[3]), method, enzyme_payoff(kind, x[4], x[2], iscall)))

@testset "Enzyme" begin
    args = bs_args()
    @test !isnothing(Base.get_extension(FinancialFFT, :FinancialFFTEnzymeCoreExt))
    for case in METHOD_CASES, (kind, toll) in case.payoffs, iscall in (true, false)
        price = enzyme_bs_pricer(case.method, Val(kind), iscall)
        g = analytic_bs_pricer(kind, iscall)
        reference = ForwardDiff.gradient(x -> g(x...), args)
        @testset "$(case.name) $kind $(iscall ? "call" : "put") $mode_name" for (mode_name, mode) in ("Forward" => Enzyme.Forward, "Reverse" => Enzyme.Reverse)
            test_within(() -> max_abs_diff(collect(only(Enzyme.gradient(mode, Const(price), args))), reference), toll.enzyme)
        end
    end
    # Scores of the densities, at a point and at an array of points, against ForwardDiff.
    y = collect(range(-0.5, 0.5, length = 11))
    @testset "$(nameof(typeof(method))) $task $mode_name" for method in (FFTDensityInverter(), CosDensityMethod()), task in ("f(y)", "density_values"), (mode_name, mode) in ("Forward" => Enzyme.Forward, "Reverse" => Enzyme.Reverse)
        merton(s) = MertonProcess(s[1], 5.0, 0.03, 0.02, Underlying(100.0, 0.01))
        score = task == "f(y)" ? (s -> log(density(merton(s), ZeroRate(0.02), method, 1.0)(0.1))) : (s -> sum(log.(FinancialFFT.density_values(density(merton(s), ZeroRate(0.02), method, 1.0), y))))
        @test only(Enzyme.gradient(mode, Const(score), [0.2])) ≈ ForwardDiff.gradient(score, [0.2]) rtol = 1e-10
    end
end
