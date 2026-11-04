
# Moves an array to the device of the mode (the CUDA extension handles the GPU modes).
adapt_array(x, _) = x
# The real floating-point type of the inputs, without their derivatives: grids, truncation ranges and the parameters of
# the methods are built in it, so that Float32 or BigFloat inputs are not promoted to Float64.
float_type(x...) = ChainRulesCore.@ignore_derivatives typeof(float(v_value_mod(sum(x))))
# A parameter of a method (a constant for differentiation) in the floating-point type FT.
to_float_type(FT, x) = ChainRulesCore.@ignore_derivatives convert(FT, v_value_mod(x))
# Relative tolerance of a computation in FT that has 1e-14 in Float64: scaled by the machine epsilon.
scaled_tolerance(FT, toll) = FT(toll) * (eps(FT) / FT(eps(Float64)))

# FFTW transforms Float32 and Float64 only: other floating-point types (BigFloat) use a radix-2 FFT (all the grids
# have 2^N points), with the twiddle factors computed in their precision.
fft_mod(x) = fft(x)
fft_mod(x::AbstractVector{Complex{T}}) where {T <: AbstractFloat} = T <: Union{Float32, Float64} ? fft(x) : generic_fft(x)
function generic_fft(x::AbstractVector{Complex{T}}) where {T}
    n = length(x)
    @assert ispow2(n) "the generic FFT needs 2^N points"
    y = collect(x)
    # bit reversal
    j = 0
    for i in 0:(n-2)
        if i < j
            y[i+1], y[j+1] = y[j+1], y[i+1]
        end
        m = n >> 1
        while m >= 1 && j & m != 0
            j ⊻= m
            m >>= 1
        end
        j |= m
    end
    half = 1
    while half < n
        for k in 0:(half-1)
            w = cispi(-T(k) / half)
            for i in (k+1):(2*half):n
                t = w * y[i+half]
                y[i+half] = y[i] - t
                y[i] += t
            end
        end
        half *= 2
    end
    return y
end

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
# One strike per column: the exponent does not depend on the strike.
strike_columns(x::Number) = x
strike_columns(x::AbstractVector) = reshape(x, 1, :)
# Sum over the first dimension: a number for a vector, one value per strike (column) for a matrix.
column_sums(terms::AbstractVector) = sum(terms)
column_sums(terms::AbstractMatrix) = vec(sum(terms; dims = 1))
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

# Put to call by parity, for the methods that price the put (COS and the density methods).
put_to_option(P, S0_adj, df, opt::VanillaPayoff) = @. ifelse(opt.isCall, P + S0_adj - opt.K * df, P)
put_to_option(P, _, df, opt::BinaryPayoff) = @. ifelse(opt.isCall, df - P, P)

# Prices computed per unit of strike: vanilla options are scaled by the strike, binaries are not.
adjust_price_adimensional(p, opt::EuropeanOption) = opt.K * p
adjust_price_adimensional(p, opt::EuropeanOptionSmile) = opt.K .* p
adjust_price_adimensional(V, ::BinaryPayoff) = V
