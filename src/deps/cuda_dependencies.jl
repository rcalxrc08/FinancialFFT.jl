using .CUDA

CUDA.allowscalar(false)
include("../methods/cuda/carrmadan.jl")
function adapt_array(x, ::FinancialMonteCarlo.AbstractCudaMode)
    return cu(x)
end
function adapt_itp(itp, ::FinancialMonteCarlo.AbstractCudaMode)
    return adapt(CuArray{Float32}, itp)
end