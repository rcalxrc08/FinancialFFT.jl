module FinancialFFTMooncakeExt

using FinancialFFT, Mooncake, FFTW
using Mooncake: CoDual, Dual, NoRData, DefaultCtx, ForwardMode, ReverseMode, primal, tangent
using Base: IEEEFloat

# Mooncake cannot differentiate through the FFTW plan (a foreigncall), so fft is a primitive.
# fft is linear, so the tangent is the fft of the tangent and the adjoint is the unnormalized inverse transform.
const ComplexVector = Vector{<:Complex{<:IEEEFloat}}

Mooncake.@is_primitive DefaultCtx ForwardMode Tuple{typeof(fft), ComplexVector}
function Mooncake.frule!!(::Dual{typeof(fft)}, x::Dual{<:ComplexVector})
    # one plan for the two transforms
    plan = plan_fft(primal(x))
    return Dual(plan * primal(x), plan * tangent(x))
end

Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(fft), ComplexVector}
function Mooncake.rrule!!(::CoDual{typeof(fft)}, x::CoDual{<:ComplexVector})
    y = fft(primal(x))
    dy = zero(y)
    dx = tangent(x)
    function fft_pullback!!(::NoRData)
        dx .+= bfft(dy)
        return NoRData(), NoRData()
    end
    return CoDual(y, dy), fft_pullback!!
end

# brfft_mod as well: the tangent is brfft of the tangent, the adjoint brfft_adjoint (the output is real).
Mooncake.@is_primitive DefaultCtx ForwardMode Tuple{typeof(FinancialFFT.brfft_mod), ComplexVector, Int}
function Mooncake.frule!!(::Dual{typeof(FinancialFFT.brfft_mod)}, x::Dual{<:ComplexVector}, n::Dual{Int})
    plan = plan_brfft(primal(x), primal(n))
    return Dual(plan * primal(x), plan * tangent(x))
end

Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(FinancialFFT.brfft_mod), ComplexVector, Int}
function Mooncake.rrule!!(::CoDual{typeof(FinancialFFT.brfft_mod)}, x::CoDual{<:ComplexVector}, n::CoDual{Int})
    y = brfft(primal(x), primal(n))
    dy = zero(y)
    dx = tangent(x)
    function brfft_pullback!!(::NoRData)
        dx .+= FinancialFFT.brfft_adjoint(dy, primal(n))
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), brfft_pullback!!
end

# The complex sqrt of Base bitcasts its argument, which Mooncake refuses to differentiate. The overlay is only seen by Mooncake.
Mooncake.@mooncake_overlay FinancialFFT.sqrt_mod(z::Complex{<:IEEEFloat}) = sqrt(sqrt(abs2(z))) * cis(angle(z) / 2)

# Mooncake ignores ChainRulesCore.@ignore_derivatives: the quantities that must stay constants go through this hook.
Mooncake.@zero_derivative DefaultCtx Tuple{typeof(FinancialFFT.ignore_derivatives_mod), IEEEFloat}

end
