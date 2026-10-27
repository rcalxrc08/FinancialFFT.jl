module FinancialFFT
using FFTW, PatternVectors, MuladdMacro, ChainRulesCore, FinancialMonteCarlo, Adapt
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
