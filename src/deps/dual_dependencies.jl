using .DualNumbers, FFTW
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