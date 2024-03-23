using .HyperDualNumbers, FFTW
import FFTW.fft!;

function fft!(x::AbstractArray{T}) where {T <: Hyper{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex, Xder1, Xder2, Xder12 = HyperDualNumbers.value.(x), HyperDualNumbers.epsilon1.(x), HyperDualNumbers.epsilon2.(x), HyperDualNumbers.epsilon12.(x)
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
    Xder1 .= planned_fft * Xder1
    Xder2 .= planned_fft * Xder2
    Xder12 .= planned_fft * Xder12
    @. x = hyper(Xcomplex, Xder1, Xder2, Xder12)
    nothing
end

function fft(x::AbstractArray{T}) where {T <: Hyper{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    y = copy(x)
    fft!(y)
    return y
end

function v_value_mod(s::Hyper)
    return s.value
end
function real_mod(x::Hyper)
    return hyper(real(x.value), real(x.epsilon1), real(x.epsilon2), real(x.epsilon12))
end

function imag_mod(x::Hyper)
    return hyper(imag_mod(x.value), imag_mod(x.epsilon1), imag_mod(x.epsilon2), imag_mod(x.epsilon12))
end