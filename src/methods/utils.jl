
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

# The unnormalised inverse DFT y_j = Σ_k h_k exp(2πi jk / n) of the Hermitian vector h (h_{n-k} = conj(h_k)) given by
# its first n/2 + 1 entries x, which is real: brfft of FFTW, which ignores the imaginary parts of x_0 and x_{n/2}.
# Other floating-point types (BigFloat) take the real part of the generic FFT of conj(h), which is conj(x) followed by
# x_{n/2-1}, …, x_1. The extensions of the AD back-ends add methods, the CUDA one through brfft of CUFFT.
brfft_mod(x, n, region = 1) = brfft(x, n, region)
brfft_mod(x::AbstractVector{Complex{T}}, n) where {T <: AbstractFloat} = T <: Union{Float32, Float64} ? brfft(x, n) : real.(generic_fft(vcat(conj.(x), view(x, div(n, 2):-1:2))))

# A plan of brfft_mod(·, n) for vectors like x, applied with *, so that the transforms of several components share one
# planning: FFTW's (CUFFT's on the device), whose planning costs 38-114 μs for 2^10-2^18 points (as much as a transform
# of 2^14); brfft_mod itself for other floats.
struct BrfftModPlan{I <: Integer}
    n::I
end
Base.:*(p::BrfftModPlan, x) = brfft_mod(x, p.n)
FFTW.mul!(y, p::BrfftModPlan, x) = copyto!(y, p * x)
plan_brfft_mod(x::AbstractVector{Complex{T}}, n) where {T <: Union{Float32, Float64}} = plan_brfft(x, n)
plan_brfft_mod(x, n) = BrfftModPlan(n)

# The adjoint of brfft_mod(·, n), for the derivatives ∂L/∂Re x + i ∂L/∂Im x of a real cotangent ȳ: the entries
# 1, …, n/2 - 1 of x stand for two entries of h, hence the weight 2; x_0 and x_{n/2} get the real values of rfft there,
# since their imaginary parts are ignored (an odd n has no x_{n/2}: the last weight is 2). The weights are a PatternVector,
# broadcast with the transform on the device too. A plan of rfft can be passed, to be reused.
function brfft_adjoint(ȳ, n, plan = plan_rfft(ȳ))
    r = plan * ȳ
    weights = ChainRulesCore.@ignore_derivatives iseven(n) ? PatternVector(length(r), PaddedFillPattern(Int8(1), Int8(2), Int8(1))) : PatternVector(length(r), InitialValuePattern(Int8(1), Int8(2)))
    return weights .* r
end
# Zygote (the rule of AbstractFFTs builds its weights on the CPU, which fails on the device).
function ChainRulesCore.rrule(::typeof(brfft_mod), x::AbstractVector, n::Integer)
    project_x = ChainRulesCore.ProjectTo(x)
    brfft_mod_pullback(ȳ) = ChainRulesCore.NoTangent(), project_x(brfft_adjoint(real(ChainRulesCore.unthunk(ȳ)), n)), ChainRulesCore.NoTangent()
    return brfft_mod(x, n), brfft_mod_pullback
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

# Reverse rules of the two recurrences (issue #47). Zygote differentiates the element reads of their loops by building a
# dense cotangent of the full length for each one, O(N²) per call; these rules are O(N), with one pass over k. With
# S = Σ_k c_k e^{ikθ} and cotangent (ā, b̄) on (Re S, Im S):
# - coefficients: c̄_re_k + i c̄_im_k = (ā + i b̄) e^{-ikθ};
# - θ: θ̄ = ā Re S' + b̄ Im S', S' = Σ_k i k c_k e^{ikθ}, so Re S' = -Σ_k k (c_re_k sin kθ + c_im_k cos kθ) and
#   Im S' = Σ_k k (c_re_k cos kθ - c_im_k sin kθ): plain sums, without the cancellation of the recurrence near θ = 0.
# cos kθ and sin kθ come by rotation, refreshed with an exact sincos every 32 terms. Only Zygote uses these rules: the
# other back-ends differentiate the loops in linear time.
clenshaw_cotangent(x) = ChainRulesCore.unthunk(x)
clenshaw_cotangent(::ChainRulesCore.AbstractZero) = false
# c_im = nothing for a real series (clenshaw_cos_sum): no imaginary coefficients, and no cotangent for them.
imaginary_coefficient(::Nothing, i) = false
imaginary_coefficient(c_im, i) = c_im[i]
imaginary_cotangent(::Nothing, T) = nothing
imaginary_cotangent(c_im, T) = similar(c_im, T)
set_imaginary_cotangent!(::Nothing, i, x) = nothing
set_imaginary_cotangent!(c̄_im, i, x) = (c̄_im[i] = x)
function clenshaw_exp_sum_pullback(c_re, c_im, θ, ā, b̄)
    n = length(c_re)
    T = typeof(float(θ * first(c_re) * ā))
    c̄_re, c̄_im = similar(c_re, T), imaginary_cotangent(c_im, T)
    s_1, c_1 = sincos(θ)
    co, si = one(T), zero(T)
    d_re = d_im = zero(T)
    for (i, k) in zip(eachindex(c_re), 0:(n-1))
        if k % 32 == 0
            si, co = sincos(k * θ)
        end
        a_k, b_k = c_re[i], imaginary_coefficient(c_im, i)
        c̄_re[i] = ā * co + b̄ * si
        set_imaginary_cotangent!(c̄_im, i, b̄ * co - ā * si)
        d_re -= k * (a_k * si + b_k * co)
        d_im += k * (a_k * co - b_k * si)
        co, si = co * c_1 - si * s_1, si * c_1 + co * s_1
    end
    return c̄_re, c̄_im, ā * d_re + b̄ * d_im
end
function ChainRulesCore.rrule(::typeof(clenshaw_exp_sum), c_re, c_im, θ, only::Val{only_real}) where {only_real}
    y = clenshaw_exp_sum(c_re, c_im, θ, only)
    function clenshaw_exp_sum_back(ȳ)
        ā, b̄ = only_real ? (clenshaw_cotangent(ȳ), false) : (clenshaw_cotangent(ȳ[1]), clenshaw_cotangent(ȳ[2]))
        c̄_re, c̄_im, θ̄ = clenshaw_exp_sum_pullback(c_re, c_im, θ, ā, b̄)
        return ChainRulesCore.NoTangent(), c̄_re, c̄_im, θ̄, ChainRulesCore.NoTangent()
    end
    return y, clenshaw_exp_sum_back
end
function ChainRulesCore.rrule(::typeof(clenshaw_exp_sum), c_re, c_im, θ)
    y, back = ChainRulesCore.rrule(clenshaw_exp_sum, c_re, c_im, θ, Val(false))
    return y, ȳ -> back(ȳ)[1:4]
end
function ChainRulesCore.rrule(::typeof(clenshaw_cos_sum), coefficients, θ)
    y = clenshaw_cos_sum(coefficients, θ)
    function clenshaw_cos_sum_back(ȳ)
        ā = clenshaw_cotangent(ȳ)
        # the real coefficients are c_re, without imaginary ones: the cotangent of c_re and θ̄ = ā Re S'
        c̄, _, θ̄ = clenshaw_exp_sum_pullback(coefficients, nothing, θ, ā, false)
        return ChainRulesCore.NoTangent(), c̄, θ̄
    end
    return y, clenshaw_cos_sum_back
end
# One strike per column: the exponent does not depend on the strike.
strike_columns(x::Number) = x
strike_columns(x::AbstractVector) = reshape(x, 1, :)
# Sum over the first dimension: a number for a vector, one value per strike (column) for a matrix.
column_sums(terms::AbstractVector) = sum(terms)
column_sums(terms::AbstractMatrix) = vec(sum(terms; dims = 1))
# Σ_j coefficient_j cos(u_j x + phase_j) for each x (a number or a vector, one per strike), as one broadcast over terms and
# strikes: the series of COS and of Lewis smiles outside SerialMode, where the coefficients cannot be indexed (GPU).
phased_cos_sum(coefficient, phase, u, x) = column_sums(@. coefficient * cos(u * $strike_columns(x) + phase))

# Lagrange polynomial through the nodes -2, …, 3 at t in [0, 1]: interpolation of order h⁶ on a uniform grid of step h.
# The six terms are paired around t = 1/2, where the products of t + 2, …, t - 3 are symmetric.
@muladd function lagrange_quintic(t, v_m2, v_m1, v_0, v_1, v_2, v_3)
    tp2, tp1, tm1, tm2, tm3 = t + 2, t + 1, t - 1, t - 2, t - 3
    a, b, c = t * tm1, tp2 * tm3, tp1 * tm2
    return b * c * (t * v_1 - tm1 * v_0) / 12 + a * b * (tm2 * v_m1 - tp1 * v_2) / 24 + a * c * (tp2 * v_3 - tm3 * v_m2) / 120
end
# Node at the left of s (a position in steps from the first node, 0-based), moved inwards within three nodes of the ends,
# so that the six nodes of the stencil exist.
lagrange_stencil(s, n) = clamp(floor(Int, v_value_mod(s)), 2, n - 4)

# Values v on the uniform grid x0 + h j, j = 0, …, length(v) - 1, interpolated at k, a number or an array, from the six
# nodes around each point. Gathers only, whose pullback is a scatter: every back-end differentiates it, on the device
# too. The grid is a constant for differentiation. The node i (0-based) is v[i + 1].
function grid_interpolate(v, x0, h, k::Number)
    s = (k - x0) / h
    i = ChainRulesCore.@ignore_derivatives lagrange_stencil(s, length(v))
    return lagrange_quintic(s - i, v[i-1], v[i], v[i+1], v[i+2], v[i+3], v[i+4])
end
function grid_interpolate(v, x0, h, k::AbstractArray)
    s = @. (k - x0) / h
    i, i_m1, i_1, i_2, i_3, i_4 = ChainRulesCore.@ignore_derivatives begin
        i = lagrange_stencil.(s, length(v))
        i, i .- 1, i .+ 1, i .+ 2, i .+ 3, i .+ 4
    end
    return lagrange_quintic.(s .- i, v[i_m1], v[i], v[i_1], v[i_2], v[i_3], v[i_4])
end
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
