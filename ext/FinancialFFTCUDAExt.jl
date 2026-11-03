module FinancialFFTCUDAExt

using FinancialFFT, FinancialMonteCarlo, CUDA, Adapt, ChainRulesCore
import FinancialFFT: adapt_array, adapt_itp

function __init__()
    CUDA.allowscalar(false)
end

# CuArray and not cu, that would convert the grids to Float32: the model parameters are usually Float64, so the
# computation would still run in Float64, with the precision of the Float32 grids.
function adapt_array(x, ::FinancialMonteCarlo.AbstractCudaMode)
    return CuArray(x)
end
adapt_array(x::Union{Number, CuArray}, ::FinancialMonteCarlo.AbstractCudaMode) = x

function adapt_itp(itp, ::CuArray)
    return adapt(CuArray, itp)
end

# PatternVectors defines the rrule of the sum of any vector, with a cotangent on the host (a PatternVector): a device
# array times it is reduced with scalar indexing in the pullbacks that follow (issue #36). The cotangent of the sum
# of a device vector stays on the device.
function ChainRulesCore.rrule(::typeof(sum), x::CuArray{T, 1}) where {T}
    sum_pullback(Δ) = (NoTangent(), fill!(similar(x), unthunk(Δ)))
    return sum(x), sum_pullback
end

end
