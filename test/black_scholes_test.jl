# Prices and number types against blsprice / blsbin from FinancialToolbox.
@testset "Black Scholes vs FinancialToolbox" begin
    args = bs_args()
    for case in METHOD_CASES, (kind, toll) in case.payoffs, iscall in (true, false)
        f = fft_bs_pricer(case.method, kind, iscall)
        g = analytic_bs_pricer(kind, iscall)
        @testset "$(case.name) $kind $(iscall ? "call" : "put")" begin
            @testset "Float64" begin
                test_within(() -> abs(f(args...) - g(args...)), toll.price)
            end
            @testset "DualNumbers ∂$name" for (i, name) in enumerate(BS_ARG_NAMES)
                dual_args = Any[args...]
                dual_args[i] = dual(args[i], 1.0)
                test_within(() -> max_abs_diff(f(dual_args...), g(dual_args...)), dual_tolerance(toll.dual, name))
            end
            @testset "HyperDualNumbers ∂S0 ∂σ" begin
                hyper_args = Any[args...]
                hyper_args[1] = hyper(BS_ARGS.S0, 1.0, 0.0, 0.0)
                hyper_args[5] = hyper(BS_ARGS.sigma, 0.0, 1.0, 0.0)
                test_within(() -> max_abs_diff(f(hyper_args...), g(hyper_args...)), toll.hyper)
            end
            @testset "TaylorSeries σ" begin
                taylor_args = Any[args...]
                taylor_args[5] = Taylor1([BS_ARGS.sigma, 1.0], 2)
                test_within(() -> max_abs_diff(f(taylor_args...), g(taylor_args...)), toll.taylor)
            end
        end
    end
end
