module FinancialFFTCUDAExt

using FinancialFFT, FinancialMonteCarlo, CUDA, ChainRulesCore
import FinancialFFT: grid_interpolate, float_type

function __init__()
    CUDA.allowscalar(false)
end

# A number on the device is interpolated as an array of one point, then summed: scalar indexing is disallowed there.
function grid_interpolate(v::CuArray, x0, h, k::Number)
    zero_on_device = ChainRulesCore.@ignore_derivatives CUDA.zeros(float_type(k), 1)
    return sum(grid_interpolate(v, x0, h, k .+ zero_on_device))
end

end
