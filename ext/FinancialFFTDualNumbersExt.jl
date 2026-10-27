module FinancialFFTDualNumbersExt

using FinancialFFT, DualNumbers, FFTW
import FFTW: fft!, fft
import FinancialFFT: v_value_mod

function v_value_mod(s::Dual)
    return s.value
end

function fft!(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex, Xder = DualNumbers.value.(x), DualNumbers.epsilon.(x)
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
    Xder .= planned_fft * Xder
    @. x = dual(Xcomplex, Xder)
    nothing
end

function fft(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    y = copy(x)
    fft!(y)
    return y
end

end
