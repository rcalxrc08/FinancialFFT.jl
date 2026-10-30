module FinancialFFTMooncakeExt

using FinancialFFT, Mooncake, FFTW
using Mooncake: CoDual, Dual, NoRData, DefaultCtx, ForwardMode, ReverseMode, primal, tangent
using Base: IEEEFloat

# Mooncake cannot differentiate through the FFTW plan (a foreigncall), so fft is a primitive.
# fft is linear, so the tangent is the fft of the tangent and the adjoint is the unnormalized inverse transform.
const ComplexVector = Vector{<:Complex{<:IEEEFloat}}

Mooncake.@is_primitive DefaultCtx ForwardMode Tuple{typeof(fft), ComplexVector}
function Mooncake.frule!!(::Dual{typeof(fft)}, x::Dual{<:ComplexVector})
    return Dual(fft(primal(x)), fft(tangent(x)))
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

# The complex sqrt of Base bitcasts its argument, which Mooncake refuses to differentiate. The overlay is only seen by Mooncake.
Mooncake.@mooncake_overlay FinancialFFT.sqrt_mod(z::Complex{<:IEEEFloat}) = sqrt(sqrt(abs2(z))) * cis(angle(z) / 2)

end
