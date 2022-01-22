using DualNumbers, HyperDualNumbers, FFTW
import FFTW.fft!;

function fft!(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex = DualNumbers.value.(x)
    Xder = epsilon.(x)
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
    Xder .= planned_fft * Xder
    @. x = dual(Xcomplex, Xder)
    nothing
end

function fft!(x::AbstractArray{T}) where {T <: Hyper{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex = HyperDualNumbers.value.(x)
    Xder1 = HyperDualNumbers.epsilon1.(x)
    Xder2 = HyperDualNumbers.epsilon2.(x)
    Xder12 = HyperDualNumbers.epsilon12.(x)
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
    Xder1 .= planned_fft * Xder1
    Xder2 .= planned_fft * Xder2
    Xder12 .= planned_fft * Xder12
    @. x = hyper(Xcomplex, Xder1, Xder2, Xder12)
    nothing
end