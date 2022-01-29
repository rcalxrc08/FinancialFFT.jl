module FinancialFFT
using Requires, FFTW # for conditional dependencies
function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("deps/cuda_dependencies.jl")
    @require DualNumbers = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74" include("deps/dual_dependencies.jl")
    @require HyperDualNumbers = "50ceba7f-c3ee-5a84-a6e8-3ad40456ec97" include("deps/hyper_dependencies.jl")
    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("deps/forwarddiff_dependencies.jl")
end
using FinancialMonteCarlo
import FinancialMonteCarlo.pricer, FinancialMonteCarlo.AbstractMethod
abstract type AbstractIntegrationMethod <: FinancialMonteCarlo.AbstractMethod end
abstract type AbstractFFTMethod <: AbstractIntegrationMethod end
abstract type AbstractIntegralMethod <: AbstractIntegrationMethod end
include("methods/utils.jl")
include("methods/charexp.jl")
include("methods/carrmadan.jl")
include("methods/carrmadan_lewis.jl")
include("methods/lewis.jl")

export pricer

end # module
