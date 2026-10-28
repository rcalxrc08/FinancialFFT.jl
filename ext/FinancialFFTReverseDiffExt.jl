module FinancialFFTReverseDiffExt

using FinancialFFT, ReverseDiff, FFTW
import FFTW: fft
import FinancialFFT: v_value_mod, ignore_derivatives_mod

function v_value_mod(s::ReverseDiff.TrackedReal)
    return ReverseDiff.value(s)
end

# ReverseDiff tracks everything, including the quantities wrapped in ChainRulesCore.@ignore_derivatives.
ignore_derivatives_mod(x::ReverseDiff.TrackedReal) = ReverseDiff.value(x)

# ReverseDiff tracks real numbers only, so the input of the fft is a vector of complex numbers with tracked parts,
# that FFTW cannot plan. The fft is recorded as a single instruction of the tape, on the real and imaginary parts;
# as in the Mooncake extension, the pullback of y = fft(x) is x̄ = bfft(ȳ), with ȳ = ∂L/∂Re(y) + i ∂L/∂Im(y).
function fft(x::AbstractVector{<:Complex{<:ReverseDiff.TrackedReal{V, D}}}) where {V, D}
    x_re, x_im = real.(x), imag.(x)
    tp = ReverseDiff.tape(x_re, x_im)
    y = fft(complex.(ReverseDiff.value(x_re), ReverseDiff.value(x_im)))
    y_re, y_im = ReverseDiff.track(real(y), D, tp), ReverseDiff.track(imag(y), D, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, fft, (x_re, x_im), (y_re, y_im))
    # the entries of a TrackedArray are tracked reals that propagate their derivative to it
    return [complex(y_re[k], y_im[k]) for k in eachindex(y)]
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fft)})
    x_re, x_im = instruction.input
    y_re, y_im = instruction.output
    x_bar = bfft(complex.(ReverseDiff.deriv(y_re), ReverseDiff.deriv(y_im)))
    ReverseDiff.increment_deriv!(x_re, real(x_bar))
    ReverseDiff.increment_deriv!(x_im, imag(x_bar))
    ReverseDiff.unseed!(y_re)
    ReverseDiff.unseed!(y_im)
    return nothing
end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fft)})
    x_re, x_im = instruction.input
    y_re, y_im = instruction.output
    ReverseDiff.pull_value!(x_re)
    ReverseDiff.pull_value!(x_im)
    y = fft(complex.(ReverseDiff.value(x_re), ReverseDiff.value(x_im)))
    ReverseDiff.value!(y_re, real(y))
    ReverseDiff.value!(y_im, imag(y))
    return nothing
end

end
