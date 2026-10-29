# CUDA is not a test dependency, since CI has no GPU: the "gpu" suite adds it to the test environment, before loading
# the packages that adding it may update.
if "gpu" in ARGS
    using Pkg
    Pkg.add("CUDA")
end

using FinancialFFT, FinancialMonteCarlo, FinancialToolbox, Test
using DualNumbers, HyperDualNumbers, TaylorSeries, ForwardDiff, ReverseDiff, Zygote, Mooncake

include("utils.jl")
include("cases.jl")

# Suites can be selected with `Pkg.test("FinancialFFT"; test_args = ["gradients"])`.
# The "gpu" suite (gpu_test.jl) needs a CUDA GPU, so it is not in SUITES and runs only when selected.
const SUITES = ["smile_structs", "extrema_computation", "black_scholes", "gradients", "other_models", "multiple_options"]
const SELECTED_SUITES = isempty(ARGS) ? SUITES : ARGS

@testset "FinancialFFT" begin
    for suite in SELECTED_SUITES
        include("$(suite)_test.jl")
    end
end
