module FinancialFFT
import FFTA
using FFTW, PatternVectors, MuladdMacro, ChainRulesCore, FinancialMonteCarlo, Adapt
import FinancialMonteCarlo: BinaryEuropeanOptionSmile, EuropeanOptionSmile
using IrrationalConstants: twoπ, invπ
include("01_hooks/abstracts.jl")
include("01_hooks/mods.jl")
include("02_range/extrema_computation_opt.jl")
include("03_utils/utils.jl")
include("04_charexp/charexp.jl")
include("05_methods/carrmadan.jl")
include("05_methods/carrmadan_lewis.jl")
include("05_methods/fft_parameters.jl")
include("05_methods/lewis.jl")
include("05_methods/cos.jl")
include("06_densities/density.jl")
include("06_densities/utils_density.jl")
include("06_densities/cos_density.jl")
include("06_densities/fft_density.jl")

export pricer

end # module
