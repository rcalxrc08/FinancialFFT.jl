using .CUDA

include("../methods/cuda/carrmadan.jl")
function adapt_array(x, ::FinancialMonteCarlo.AbstractCudaMode)
    return cu(x)
end
function adapt_itp(itp, ::FinancialMonteCarlo.AbstractCudaMode)
    return adapt(CuArray{Float32}, itp)
end
CUDA.allowscalar(false)

Base.BroadcastStyle(a::Broadcast.ArrayStyle{AlternateVector{T}}, ::CUDA.CuArrayStyle{0}) where {T} = a
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternateVector{T}}, a::CUDA.CuArrayStyle{N}) where {T, N} = a

Base.BroadcastStyle(a::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, ::CUDA.CuArrayStyle{0}) where {T} = a
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, a::CUDA.CuArrayStyle{N}) where {T, N} = a