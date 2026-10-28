module FinancialFFTCUDAExt

using FinancialFFT, FinancialMonteCarlo, CUDA, Adapt
import FinancialFFT: adapt_array, adapt_itp

function __init__()
    CUDA.allowscalar(false)
end

# CuArray and not cu, that would convert the grids to Float32: the model parameters are usually Float64, so the
# computation would still run in Float64, with the precision of the Float32 grids.
function adapt_array(x, ::FinancialMonteCarlo.AbstractCudaMode)
    return CuArray(x)
end

function adapt_itp(itp, ::CuArray)
    return adapt(CuArray, itp)
end

end
