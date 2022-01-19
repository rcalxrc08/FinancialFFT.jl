using DualNumbers, HyperDualNumbers, FFTW
import FFTW.fft!;

function fft!(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex_ = DualNumbers.value.(x)
    Xder = epsilon.(x)
    fft!(Xcomplex_)
    fft!(Xder)
    @. x = dual(Xcomplex, Xder)
    nothing
end

function fft!(x::AbstractArray{T}) where {T <: Hyper{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex = HyperDualNumbers.value.(x)
    Xder1 = HyperDualNumbers.epsilon1.(x)
    Xder2 = HyperDualNumbers.epsilon2.(x)
    Xder12 = HyperDualNumbers.epsilon12.(x)
    fft!(Xcomplex)
    fft!(Xder1)
    fft!(Xder2)
    fft!(Xder12)
    @. x = hyper(Xcomplex, Xder1, Xder2, Xder12)
    nothing
end