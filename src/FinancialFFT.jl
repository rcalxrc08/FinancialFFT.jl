module FinancialFFT
using Requires, FFTW # for conditional dependencies
function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("deps/cuda_dependencies.jl")
    @require DualNumbers = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74" include("deps/dual_dependencies.jl")
    @require HyperDualNumbers = "50ceba7f-c3ee-5a84-a6e8-3ad40456ec97" include("deps/hyper_dependencies.jl")
    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("deps/forwarddiff_dependencies.jl")
    @require TaylorSeries = "6aa5eb33-94cf-58f4-a9d0-e4b2c4fc25ea" include("deps/taylorseries_dependencies.jl")
end

include("methods/abstracts.jl")
include("methods/extrema_computation_opt.jl")
include("methods/utils.jl")
include("methods/charexp.jl")
include("methods/carrmadan.jl")
include("methods/carrmadan_lewis.jl")
include("methods/lewis.jl")
include("methods/cos.jl")
include("methods/density.jl")

export pricer

end # module
