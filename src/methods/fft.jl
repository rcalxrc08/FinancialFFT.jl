using DualNumbers, FFTW
import FFTW.fft;
import FFTW.fft!;

function fft(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex = DualNumbers.value.(x)
    Xder = epsilon.(x)
    fft!(Xcomplex)
    fft!(Xder)
    Yout = [T(real1, epsilon1) for (real1, epsilon1) in zip(Xcomplex, Xder)]

    return Yout
end

#In place fft does not exist for 'real' numbers.
function fft(x::AbstractArray{T}) where {T <: Dual{num}} where {num <: Number}
    Xcomplex = DualNumbers.value.(x)
    pritln("ciao")
    Xder = epsilon.(x)
    x_ = fft(Xcomplex)
    y_ = fft(Xder)
    Yout = [Dual(real1, epsilon1) for (real1, epsilon1) in zip(x_, y_)]

    return Yout
end

function fft!(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex_ = DualNumbers.value.(x)
    Xder = epsilon.(x)
    fft!(Xcomplex_)
    fft!(Xder)
    x .= [T(real1, epsilon1) for (real1, epsilon1) in zip(Xcomplex_, Xder)]
    nothing
end

#In place fft does not exist for 'real' numbers.
function fft!(x::AbstractArray{T}) where {T <: Dual{num}} where {num <: Number}
    Xcomplex = DualNumbers.value.(x)
    Xder = epsilon.(x)
    x_ = fft(Xcomplex)
    y_ = fft(Xder)

    x .= [Dual(real1, epsilon1) for (real1, epsilon1) in zip(x_, y_)]
    nothing
end

function real_mod(x::Hyper)
    return hyper(real(x.value), real(x.epsilon1), real(x.epsilon2), real(x.epsilon12))
end
real_mod(x) = real(x)

Base.hash(x::Dual) = Base.hash(Base.hash("r") - Base.hash(x.value) - Base.hash("i") - Base.hash(x.epsilon))

function fft!(x::AbstractArray{T}) where {T <: Hyper{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex = HyperDualNumbers.value.(x)
    Xder1 = HyperDualNumbers.epsilon1.(x)
    Xder2 = HyperDualNumbers.epsilon2.(x)
    Xder12 = HyperDualNumbers.epsilon12.(x)
    val = fft(Xcomplex)
    der1 = fft(Xder1) #* length(Xder1)
    der2 = fft(Xder2) #* length(Xder1)
    der12 = fft(Xder12)
    x .= [hyper(x, y_1, y_2, z) for (x, y_1, y_2, z) in zip(val, der1, der2, der12)]
    nothing
end