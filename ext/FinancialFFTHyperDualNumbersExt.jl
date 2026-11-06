module FinancialFFTHyperDualNumbersExt

using FinancialFFT, HyperDualNumbers, FFTW
import FFTW: fft!, fft
import FinancialFFT: v_value_mod, real_mod, imag_mod, sincos_mod, brfft_mod, plan_brfft_mod

# HyperDualNumbers has no sincos. f(x) = f(v) + f'(v) (ε₁ + ε₂ + ε₁₂) + f''(v) ε₁ε₂, and the first two derivatives of
# sin and cos are again ±sin and ±cos of the value: one sincos of the value gives the two.
function sincos_mod(x::Hyper)
    s, c = sincos(x.value)
    ε₁, ε₂, ε₁₂ = x.epsilon1, x.epsilon2, x.epsilon12
    ε₁ε₂ = ε₁ * ε₂
    return hyper(s, c * ε₁, c * ε₂, c * ε₁₂ - s * ε₁ε₂), hyper(c, -s * ε₁, -s * ε₂, -s * ε₁₂ - c * ε₁ε₂)
end

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

# brfft is linear: one transform per component, with one plan, whose output is real.
function brfft_mod(x::AbstractVector{T}, n::Integer) where {T <: Hyper{<:Complex}}
    value = HyperDualNumbers.value.(x)
    plan = plan_brfft_mod(value, n)
    return hyper.(plan * value, plan * HyperDualNumbers.epsilon1.(x), plan * HyperDualNumbers.epsilon2.(x), plan * HyperDualNumbers.epsilon12.(x))
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

end
