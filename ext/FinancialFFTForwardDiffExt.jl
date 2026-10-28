module FinancialFFTForwardDiffExt

using FinancialFFT, ForwardDiff, FFTW
import FFTW: fft!, fft
import FinancialFFT: v_value_mod

function v_value_mod(s::ForwardDiff.Dual)
    return s.value
end

dual_component(x, k) = k == 0 ? ForwardDiff.value(x) : ForwardDiff.partials(x, k)

# The fft is linear, so it maps the values and each partial separately: they are the columns of one matrix, transformed
# by a single batched fft. Everything else is a broadcast, so that x can also be a GPU array.
function fft!(x::AbstractVector{Complex{ForwardDiff.Dual{Tg, T, N}}}) where {Tg, T <: Real, N}
    components = (0:N)'
    M = @. complex(dual_component(real(x), components), dual_component(imag(x), components))
    fft!(M, 1)
    to_dual(value, partials...) = complex(ForwardDiff.Dual{Tg}(real(value), real.(partials)), ForwardDiff.Dual{Tg}(imag(value), imag.(partials)))
    x .= to_dual.(ntuple(k -> view(M, :, k), Val(N + 1))...)
    return x
end

function fft(x::AbstractVector{Complex{ForwardDiff.Dual{Tg, T, N}}}) where {Tg, T <: Real, N}
    return fft!(copy(x))
end

end
