module FinancialFFT
using Requires # for conditional dependencies
function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("deps/cuda_dependencies.cujl")
end
using FinancialMonteCarlo
import FinancialMonteCarlo.pricer, FinancialMonteCarlo.AbstractMethod
abstract type AbstractIntegrationMethod <: FinancialMonteCarlo.AbstractMethod end
abstract type AbstractFFTMethod <: AbstractIntegrationMethod end
abstract type AbstractIntegralMethod <: AbstractIntegrationMethod end
include("methods/utils.jl")
include("methods/charexp.jl")
include("methods/fft.jl")
include("methods/carrmadan.jl")
include("methods/carrmadan_lewis.jl")
include("methods/lewis.jl")

export pricer

end # module
