"""
    FFTDensityInverter(N = 14)

Density of the log-return `log(S_T / S_0)` recovered by inverting the characteristic function with a single FFT on
`2^N` points, see [`density`](@ref). The grid spans the range of `CosMethod`, from a Chernoff bound on the tails, so,
it needs no range parameter for any model, and it narrows with the
horizon: the short steps of a Vibrato or likelihood ratio estimator keep the resolution of long ones. Between the
nodes the density is interpolated by cubic Lagrange polynomials, written with broadcasts and gathers only, so that
every AD back-end, Zygote included, differentiates it, on the CPU and on the GPU.

# Arguments
- `N`: base-2 logarithm of the number of grid points (`N > 2`).
"""
struct FFTDensityInverter{num_1 <: Integer} <: AbstractIntegralMethod
    N::num_1
    function FFTDensityInverter(N::num_1 = 14) where {num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        return new{num_1}(N)
    end
end
export FFTDensityInverter

"""
    FFTDensity

Density of the log-return of [`FFTDensityInverter`](@ref), returned by [`density`](@ref): its `values` on the grid
`a, a + dx, …, b` of `2^N` points, which stay on the device of the mode that computed them, and are interpolated by
cubic Lagrange polynomials on four nodes, with an error of order `dx⁴`. Like the grid values, the interpolated density
is floored at the machine epsilon, so that its logarithm, and a score, stay finite in the tails.

Calling it on a number evaluates it on the CPU; [`FinancialFFT.density_values`](@ref) evaluates it at an array of
points, on the GPU too. `adapt(Array, f)` moves a density computed on the GPU to the CPU.
"""
struct FFTDensity{num_1, num_2} <: AbstractDensity
    values::num_1
    a::num_2
    b::num_2
    dx::num_2
end
Adapt.@adapt_structure FFTDensity

function density(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FFTDensityInverter, T, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    N = method.N
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    # the range is a constant for differentiation: a grid moving with the parameters would add the derivative of the
    # interpolation error to the scores (Mooncake ignores @ignore_derivatives, hence the hook)
    a, b = ChainRulesCore.@ignore_derivatives ignore_derivatives_mod.(v_value_mod.(compute_chernorff_limits(mcProcess, rT, dT, T)))
    n = ChainRulesCore.@ignore_derivatives 2^N
    A = (b - a) / 2
    # the grid of fft_density_values is centred at zero: the shift moves the centre of the range there
    values = fft_density_values(mcProcess, zero_rate, N, A, -(a + b) / 2, T, mode)
    dx = 2 * A / n
    return FFTDensity(values, a, a + (n - 1) * dx, dx)
end

# Lagrange polynomial through (-1, v_m1), (0, v_0), (1, v_1), (2, v_2), at t in [0, 1].
@muladd function lagrange_cubic(t, v_m1, v_0, v_1, v_2)
    tp1, tm1, tm2 = t + 1, t - 1, t - 2
    return (tp1 * tm2 * (tm1 * v_0 - t * v_1)) / 2 + (t * tm1 * (tp1 * v_2 - tm2 * v_m1)) / 6
end

# Index of the node at the left of s (a position in units of dx from the first node), moved inwards at the ends so that
# the four nodes of the stencil exist; then the floored interpolation.
stencil_index(s, n) = clamp(floor(Int, v_value_mod(s)), 1, n - 3)
function interpolate_density(t, v_m1, v_0, v_1, v_2)
    value = lagrange_cubic(t, v_m1, v_0, v_1, v_2)
    return density_floor(value, eps(float_type(value)))
end

function (f::FFTDensity)(y)
    v = f.values
    s = (clamp(y, f.a, f.b) - f.a) / f.dx
    i = ChainRulesCore.@ignore_derivatives stencil_index(s, length(v))
    # 1-based: the node i (0-based) is v[i + 1]
    value = interpolate_density(s - i, v[i], v[i+1], v[i+2], v[i+3])
    is_inside = f.a <= v_value_mod(y) <= f.b
    return ifelse(is_inside, value, zero(value))
end

"""
    FinancialFFT.density_values(f, y)

Values of the density `f` (an [`FinancialFFT.AbstractDensity`](@ref)) at the array of log-returns `y`, with the shape
of `y`. For `FFTDensity` and `CosDensity` it uses broadcasts, gathers and reductions only: it runs on the device of
`y` and of the density, and Zygote differentiates it there too.
"""
function density_values(f::FFTDensity, y::AbstractArray)
    v = f.values
    a = f.a
    s = @. (clamp(y, a, f.b) - a) / f.dx
    i, i_1, i_2, i_3 = ChainRulesCore.@ignore_derivatives begin
        i = stencil_index.(s, length(v))
        i, i .+ 1, i .+ 2, i .+ 3
    end
    is_inside = ChainRulesCore.@ignore_derivatives @. a <= v_value_mod(y) <= f.b
    # gathers: indexing with arrays of indices runs on the device, and its pullback is a scatter
    value = interpolate_density.(s .- i, v[i], v[i_1], v[i_2], v[i_3])
    return @. ifelse(is_inside, value, zero(value))
end
# beyond the range the cosine series is periodic: it is masked, as in the callable
function density_values(f::CosDensity, y::AbstractArray)
    is_inside = ChainRulesCore.@ignore_derivatives @. f.a <= v_value_mod(y) <= f.b
    value = cos_density_values(f, y)
    return @. ifelse(is_inside, value, zero(value))
end
density_values(f::AbstractDensity, y::AbstractArray) = map(f, y)
pricing_density_values(f::FFTDensity, y, _) = density_values(f, y)
