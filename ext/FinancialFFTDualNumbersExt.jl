module FinancialFFTDualNumbersExt

using FinancialFFT, DualNumbers, FFTW
import FFTW: fft!, fft
import FinancialFFT: v_value_mod, sincos_mod, brfft_mod, plan_brfft_mod

# DualNumbers has no sincos. sin and cos of a dual each need both sin and cos of its value (one for the value, the other
# for the derivative): one sincos of the value gives the two.
function sincos_mod(x::Dual)
    s, c = sincos(x.value)
    return Dual(s, c * x.epsilon), Dual(c, -s * x.epsilon)
end

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

# brfft is linear: one transform for the value and one for the derivative, with one plan, whose outputs are real.
function brfft_mod(x::AbstractVector{T}, n::Integer) where {T <: Dual{<:Complex}}
    value, epsilon = DualNumbers.value.(x), DualNumbers.epsilon.(x)
    plan = plan_brfft_mod(value, n)
    return dual.(plan * value, plan * epsilon)
end

end
