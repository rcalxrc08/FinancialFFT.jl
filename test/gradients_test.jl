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
