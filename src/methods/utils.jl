
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

# Σ_k (c_re_k + i c_im_k) exp(i k θ), k = 0, 1, …, as (real part, imaginary part), by the Clenshaw recurrence
# b_k = C_k + 2 cos(θ) b_{k+1} - b_{k+2}, S = b_0 - exp(-iθ) b_1. Near θ = 0 (or π) the b_k grow like the number of
# terms and S cancels, so there it runs in Reinsch's form, on d_k = b_k - σ b_{k+1} with σ = 1 (or -1), and with
# g = σ - cos(θ) = 2 sin²(θ/2) (or -2 cos²(θ/2)) computed without cancellation:
# d_k = C_k - 2 g b_{k+1} + σ d_{k+1}, b_k = d_k + σ b_{k+1}, S = d_0 + (g + i sin(θ)) b_1.
# Reinsch's form has a longer chain of dependent operations (about 1.8 times slower), hence the plain one for
# |cos(θ)| <= 0.95, where its error is that of a sum (COS, densities). On the coefficients of Lewis (θ = h c, close to 0)
# the plain one errs by up to 1e-7 near the money, Reinsch's by 1e-13.
# With Val(true) only the real part is returned: the imaginary recurrence still runs, since b_1 enters it.
function clenshaw_exp_sum(c_re, c_im, θ, ::Val{only_real} = Val(false)) where {only_real}
    s_h, c_h = sincos_mod(θ / 2)
    s = 2 * s_h * c_h
    c = (c_h - s_h) * (c_h + s_h)
    zero_typed = zero(c * first(c_re) * first(c_im))
    if abs(v_value_mod(c)) <= 19 // 20
        t = 2 * c
        b_1r = b_2r = b_1i = b_2i = zero_typed
        for k in lastindex(c_re):-1:(firstindex(c_re)+1)
            b_1r, b_2r = @muladd(t * b_1r + (c_re[k] - b_2r)), b_1r
            b_1i, b_2i = @muladd(t * b_1i + (c_im[k] - b_2i)), b_1i
        end
        S_r = @muladd(first(c_re) + c * b_1r - b_2r - s * b_1i)
        only_real && return S_r
        return S_r, @muladd(first(c_im) + c * b_1i - b_2i + s * b_1r)
    end
    near_zero = v_value_mod(c) > 0
    σ = ifelse(near_zero, 1, -1)
    g = ifelse(near_zero, 2 * s_h^2, -2 * c_h^2)
    γ = -2 * g
    b_r = d_r = b_i = d_i = zero_typed
    for k in lastindex(c_re):-1:(firstindex(c_re)+1)
        d_r = @muladd(c_re[k] + γ * b_r + σ * d_r)
        b_r = @muladd(d_r + σ * b_r)
        d_i = @muladd(c_im[k] + γ * b_i + σ * d_i)
        b_i = @muladd(d_i + σ * b_i)
    end
    S_r = @muladd(first(c_re) + γ * b_r + σ * d_r + g * b_r - s * b_i)
    only_real && return S_r
    return S_r, @muladd(first(c_im) + γ * b_i + σ * d_i + g * b_i + s * b_r)
end
# The same for real coefficients: Σ_k c_k cos(k θ).
function clenshaw_cos_sum(coefficients, θ)
    s_h, c_h = sincos_mod(θ / 2)
    c = (c_h - s_h) * (c_h + s_h)
    zero_typed = zero(c * first(coefficients))
    if abs(v_value_mod(c)) <= 19 // 20
        t = 2 * c
        b_1 = b_2 = zero_typed
        for k in lastindex(coefficients):-1:(firstindex(coefficients)+1)
            b_1, b_2 = @muladd(t * b_1 + (coefficients[k] - b_2)), b_1
        end
        return @muladd(first(coefficients) + c * b_1 - b_2)
    end
    near_zero = v_value_mod(c) > 0
    σ = ifelse(near_zero, 1, -1)
    g = ifelse(near_zero, 2 * s_h^2, -2 * c_h^2)
    γ = -2 * g
    b = d = zero_typed
    for k in lastindex(coefficients):-1:(firstindex(coefficients)+1)
        d = @muladd(coefficients[k] + γ * b + σ * d)
        b = @muladd(d + σ * b)
    end
    return @muladd(first(coefficients) + γ * b + σ * d + g * b)
end
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
