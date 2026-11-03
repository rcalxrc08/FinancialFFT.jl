
real_mod(x) = real(x)
imag_mod(x) = imag(x)
# Mooncake cannot differentiate the complex sqrt of Base: its extension overlays this hook.
sqrt_mod(x) = sqrt(x)
# Base defines cis for real numbers only: the TaylorSeries extension overloads this hook.
cis_mod(x) = cis(x)
# The abs of a Taylor series refuses a constant term of zero, which a density has at some nodes of its tails: the
# TaylorSeries extension overloads this hook.
abs_mod(x) = abs(x)
# DualNumbers and HyperDualNumbers have no sincos: their extensions overload this hook.
sincos_mod(x) = sincos(x)
adapt_array(x, _) = x
# Fft of each column. The AD back-ends overload the fft of a vector, hence the fallback; Mooncake sees plain complex
# numbers, and reaches it through an overlay.
fft_columns(x::AbstractVector) = fft(x)
fft_columns(M::AbstractMatrix) = fft_each_column(M)
fft_columns(M::AbstractMatrix{<:Complex{<:AbstractFloat}}) = fft(M, 1)
fft_each_column(M) = reduce(hcat, [fft(M[:, j]) for j in axes(M, 2)])
# One strike per column: the exponent does not depend on the strike.
strike_columns(x::Number) = x
strike_columns(x::AbstractVector) = reshape(x, 1, :)
# Quantities that are constants for differentiation (truncation ranges, grids). ReverseDiff does not honor
# ChainRulesCore.@ignore_derivatives, so its extension strips the tracking here.
ignore_derivatives_mod(x) = x
# Moves an interpolation to the device of the points where it is evaluated (the CUDA extension handles CuArrays).
function adapt_itp(itp, _)
    return itp
end
include("smile.jl")
const VanillaPayoff = Union{EuropeanOption, EuropeanOptionSmile}
const BinaryPayoff = Union{BinaryEuropeanOption, BinaryEuropeanOptionSmile}
const SmilePayoff = Union{EuropeanOptionSmile, BinaryEuropeanOptionSmile}
const FourierPayoff = Union{VanillaPayoff, BinaryPayoff}
function call_to_put(C, S0_adj, df, opt::EuropeanOption)
    adj = opt.K * df - S0_adj
    return ifelse(opt.isCall, C, C + adj)
end

function call_to_put(C, S0_adj, df, opt::EuropeanOptionSmile)
    return @. ifelse(opt.isCall, C, C - S0_adj + opt.K * df)
end

function call_to_put(C, _, df, opt::BinaryEuropeanOption)
    return ifelse(opt.isCall, C, df - C)
end

function call_to_put(C, _, df, opt::BinaryEuropeanOptionSmile)
    return @. ifelse(opt.isCall, C, df - C)
end
