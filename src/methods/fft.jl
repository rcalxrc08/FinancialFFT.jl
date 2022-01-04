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
    Xder = epsilon.(x)
    x_ = fft(Xcomplex)
    y_ = fft(Xder)
    Yout = [Dual(real1, epsilon1) for (real1, epsilon1) in zip(x_, y_)]

    return Yout
end

function fft!(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex = DualNumbers.value.(x)
    Xder = epsilon.(x)
    fft!(Xcomplex)
    fft!(Xder)
    x .= [T(real1, epsilon1) for (real1, epsilon1) in zip(Xcomplex, Xder)]
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
