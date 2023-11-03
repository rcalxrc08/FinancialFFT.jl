using .CUDA

function midpoint_definite_integral_cu(f, xmin, xmax, N)
    x = cu(collect(range(xmin, length = N, stop = xmax)))
    dx = (xmax - xmin) / (N - 1)
    # sum_ = zero(typeof(f(xmin))) * dx * 0
    # for x_ in x
    #     x_ != 0 && !isnan(f(x_)) && (sum_ += f(x_) * dx)
    # end
    return sum(f.(x) .* dx)
end

include("../methods/cuda/carrmadan.jl")
include("../methods/cuda/density.jl")
include("../methods/cuda/lewis.jl")
include("../methods/cuda/carrmadan_lewis.jl")

CUDA.allowscalar(false)

Base.BroadcastStyle(a::Broadcast.ArrayStyle{AlternateVector{T}}, ::CUDA.CuArrayStyle{0}) where {T} = a
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternateVector{T}}, a::CUDA.CuArrayStyle{N}) where {T, N} = a

Base.BroadcastStyle(a::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, ::CUDA.CuArrayStyle{0}) where {T} = a
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, a::CUDA.CuArrayStyle{N}) where {T,N} = a