module FinancialFFTReverseDiffExt

using FinancialFFT, ReverseDiff, FFTW
import FFTW: fft
import FinancialFFT: v_value_mod, brfft_mod, brfft_adjoint

function v_value_mod(s::ReverseDiff.TrackedReal)
    return ReverseDiff.value(s)
end

# ReverseDiff tracks real numbers only, so the input of the fft is a vector of complex numbers with tracked parts,
# that FFTW cannot plan. The fft is recorded as a single instruction of the tape, on the real and imaginary parts;
# as in the Mooncake extension, the pullback of y = fft(x) is x̄ = bfft(ȳ), with ȳ = ∂L/∂Re(y) + i ∂L/∂Im(y).
function fft(x::AbstractVector{<:Complex{<:ReverseDiff.TrackedReal{V, D}}}) where {V, D}
    x_re, x_im = real.(x), imag.(x)
    tp = ReverseDiff.tape(x_re, x_im)
    x_value = complex.(ReverseDiff.value(x_re), ReverseDiff.value(x_im))
    # the plans are the cache of the instruction: a compiled tape reuses them at every run
    plans = (plan_fft(x_value), plan_bfft(x_value))
    y = plans[1] * x_value
    y_re, y_im = ReverseDiff.track(real(y), D, tp), ReverseDiff.track(imag(y), D, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, fft, (x_re, x_im), (y_re, y_im), plans)
    # the entries of a TrackedArray are tracked reals that propagate their derivative to it
    return [complex(y_re[k], y_im[k]) for k in eachindex(y)]
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fft)})
    x_re, x_im = instruction.input
    y_re, y_im = instruction.output
    x_bar = instruction.cache[2] * complex.(ReverseDiff.deriv(y_re), ReverseDiff.deriv(y_im))
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
    y = instruction.cache[1] * complex.(ReverseDiff.value(x_re), ReverseDiff.value(x_im))
    ReverseDiff.value!(y_re, real(y))
    ReverseDiff.value!(y_im, imag(y))
    return nothing
end

# brfft, recorded the same way, with the adjoint of brfft_mod; its output is real. The cache holds n and the plans.
function brfft_mod(x::AbstractVector{<:Complex{<:ReverseDiff.TrackedReal{V, D}}}, n::Integer) where {V, D}
    x_re, x_im = real.(x), imag.(x)
    tp = ReverseDiff.tape(x_re, x_im)
    x_value = complex.(ReverseDiff.value(x_re), ReverseDiff.value(x_im))
    plan_b = plan_brfft(x_value, n)
    y_value = plan_b * x_value
    y = ReverseDiff.track(y_value, D, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, brfft_mod, (x_re, x_im), y, (n, plan_b, plan_rfft(y_value)))
    # the output is real: the TrackedArray itself, whose broadcasts are recorded as one instruction each
    return y
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(brfft_mod)})
    x_re, x_im = instruction.input
    y = instruction.output
    n, _, plan_r = instruction.cache
    x_bar = brfft_adjoint(ReverseDiff.deriv(y), n, plan_r)
    ReverseDiff.increment_deriv!(x_re, real(x_bar))
    ReverseDiff.increment_deriv!(x_im, imag(x_bar))
    ReverseDiff.unseed!(y)
    return nothing
end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(brfft_mod)})
    x_re, x_im = instruction.input
    ReverseDiff.pull_value!(x_re)
    ReverseDiff.pull_value!(x_im)
    ReverseDiff.value!(instruction.output, instruction.cache[2] * complex.(ReverseDiff.value(x_re), ReverseDiff.value(x_im)))
    return nothing
end

end
