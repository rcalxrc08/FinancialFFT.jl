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
include("context.jl")

# Suites can be selected with `Pkg.test("FinancialFFT"; test_args = ["gradients"])`.
# The "gpu" suite (gpu_test.jl) needs a CUDA GPU, so it is not in SUITES and runs only when selected.
const SUITES = ["extrema_computation", "fft_parameters", "black_scholes", "gradients", "enzyme", "other_models", "multiple_options", "density", "types", "aqua"]
const SELECTED_SUITES = isempty(ARGS) ? SUITES : ARGS

@testset "FinancialFFT" begin
    for suite in SELECTED_SUITES
        # the testsets report only at the end: log each suite so a hung one shows up in the log
        @info "Running $(suite)_test.jl"
        flush(stderr)
        elapsed = @elapsed include("$(suite)_test.jl")
        @info "Finished $(suite)_test.jl in $(round(elapsed, digits = 1)) s"
        flush(stderr)
    end
end
