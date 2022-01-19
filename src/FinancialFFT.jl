module FinancialFFT

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
