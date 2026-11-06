module FinancialFFT
import FFTA
using FFTW, PatternVectors, MuladdMacro, ChainRulesCore, FinancialMonteCarlo, Adapt
import FinancialMonteCarlo: BinaryEuropeanOptionSmile, EuropeanOptionSmile
using IrrationalConstants: twoπ, invπ
include("methods/abstracts.jl")
include("methods/mods.jl")
include("methods/extrema_computation_opt.jl")
include("methods/utils.jl")
include("methods/charexp.jl")
include("methods/carrmadan.jl")
include("methods/carrmadan_lewis.jl")
include("methods/fft_parameters.jl")
include("methods/lewis.jl")
include("methods/cos.jl")
include("methods/density.jl")
include("methods/utils_density.jl")
include("methods/cos_density.jl")
include("methods/fft_density.jl")

export pricer

end # module
