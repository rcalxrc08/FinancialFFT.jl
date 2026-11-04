module FinancialFFTCUDAExt

using FinancialFFT, FinancialMonteCarlo, CUDA, ChainRulesCore
import FinancialFFT: adapt_array, grid_interpolate, float_type

function __init__()
    CUDA.allowscalar(false)
end

# CuArray and not cu, that would convert the grids to Float32: the model parameters are usually Float64, so the
# computation would still run in Float64, with the precision of the Float32 grids.
function adapt_array(x, ::FinancialMonteCarlo.AbstractCudaMode)
    return CuArray(x)
end
adapt_array(x::Union{Number, CuArray}, ::FinancialMonteCarlo.AbstractCudaMode) = x

# A number on the device is interpolated as an array of one point, then summed: scalar indexing is disallowed there.
function grid_interpolate(v::CuArray, x0, h, k::Number)
    zero_on_device = ChainRulesCore.@ignore_derivatives CUDA.zeros(float_type(k), 1)
    return sum(grid_interpolate(v, x0, h, k .+ zero_on_device))
end

end
