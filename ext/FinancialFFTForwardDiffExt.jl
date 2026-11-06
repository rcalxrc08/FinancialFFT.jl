module FinancialFFTForwardDiffExt

using FinancialFFT, ForwardDiff, FFTW
import FFTW: fft!, fft
import FinancialFFT: v_value_mod, brfft_mod

# Nested duals (Hessians): the value is the one of the innermost dual.
function v_value_mod(s::ForwardDiff.Dual)
    return v_value_mod(s.value)
end

dual_component(x, k) = k == 0 ? ForwardDiff.value(x) : ForwardDiff.partials(x, k)
# Index of the component along a new last dimension.
component_indices(::AbstractVector, N) = (0:N)'
component_indices(x::AbstractArray, N) = reshape(0:N, ntuple(_ -> 1, ndims(x))..., N + 1)

# The fft is linear, so it maps the values and each partial separately: they are stacked along a new last dimension,
# transformed by a single batched fft. Everything else is a broadcast, so that x can also be a GPU array. The components
# of nested duals (Hessians) are duals: their array goes through this method again, one dimension more per level.
function fft!(x::AbstractArray{Complex{ForwardDiff.Dual{Tg, T, N}}}, region = 1:ndims(x)) where {Tg, T <: Real, N}
    components = component_indices(x, N)
    M = @. complex(dual_component(real(x), components), dual_component(imag(x), components))
    FinancialFFT.fft_mod!(M, region)
    to_dual(value, partials...) = complex(ForwardDiff.Dual{Tg}(real(value), real.(partials)), ForwardDiff.Dual{Tg}(imag(value), imag.(partials)))
    x .= to_dual.(ntuple(k -> selectdim(M, ndims(M), k), Val(N + 1))...)
    return x
end

function fft(x::AbstractArray{Complex{ForwardDiff.Dual{Tg, T, N}}}, region = 1:ndims(x)) where {Tg, T <: Real, N}
    return fft!(copy(x), region)
end

# brfft is linear too: the components are stacked along a new last dimension and transformed by one batched brfft
# along the first; its output is real. Nested duals recurse as in fft!.
function brfft_mod(x::AbstractArray{Complex{ForwardDiff.Dual{Tg, T, N}}}, n::Integer, region = 1) where {Tg, T <: Real, N}
    components = component_indices(x, N)
    M = @. complex(dual_component(real(x), components), dual_component(imag(x), components))
    Y = brfft_mod(M, n, region)
    to_dual(value, partials...) = ForwardDiff.Dual{Tg}(value, partials)
    return to_dual.(ntuple(k -> selectdim(Y, ndims(Y), k), Val(N + 1))...)
end

end
