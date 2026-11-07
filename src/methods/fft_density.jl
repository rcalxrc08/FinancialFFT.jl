"""
    FFTDensityInverter(N = 14, M = 2^10; mode = SerialMode())

Density of the log-return `log(S_T / S_0)` recovered by inverting the characteristic function with a single
complex-to-real FFT on `2^N` points, see [`density`](@ref): the density is real, so the characteristic function is
evaluated on the `2^(N-1) + 1` non-negative frequencies only. The grid spans the range of `CosMethod`, from a Chernoff bound on the tails, so it
needs no range parameter for any model, and it narrows with the horizon: the short steps of a Vibrato or likelihood
ratio estimator keep the resolution of long ones. Between the nodes the density is interpolated by cubic Lagrange
polynomials, written with broadcasts and gathers only, so that every AD back-end, Zygote included, differentiates it,
on the CPU and on the GPU.

As a pricer, it integrates the put against this density by Simpson's rule on `M` intervals of log-moneyness ending at
the strike, so that the kink of the payoff is an endpoint, and recovers the call by parity, as
[`CosDensityMethod`](@ref) does. Supports `EuropeanOption`, `BinaryEuropeanOption` and their smiles: the strikes of a
smile share the density, and their prices are the ones of the single options.

# Arguments
- `N`: base-2 logarithm of the number of grid points (`N > 2`).
- `M`: number of Simpson intervals for each strike (even, `M > 2`); the density does not use it.
- `mode` (keyword): the execution mode, `FinancialMonteCarlo.SerialMode()` (default) or `FinancialMonteCarlo.CudaMode()` with CUDA.jl loaded; `FinancialMonteCarlo.with_mode(method, mode)` changes it.
"""
struct FFTDensityInverter{num_1 <: Integer, num_2 <: Integer, modeType <: FinancialMonteCarlo.AbstractMode} <: AbstractDensityMethod
    N::num_1
    M::num_2
    mode::modeType
    function FFTDensityInverter(N::num_1, M::num_2, mode::modeType) where {num_1 <: Integer, num_2 <: Integer, modeType <: FinancialMonteCarlo.AbstractMode}
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        ChainRulesCore.@ignore_derivatives @assert(M > 2 && iseven(M), "M must be even and greater than 2")
        return new{num_1, num_2, modeType}(N, M, mode)
    end
end
FFTDensityInverter(N::Integer = 14, M::Integer = 2^10; mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = FFTDensityInverter(N, M, mode)
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

function density(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::FFTDensityInverter, T)
    mode = FinancialMonteCarlo.execution_mode(method)
    N = method.N
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    # the range is a constant for differentiation: a grid moving with the parameters would add the derivative of the
    # interpolation error to the scores (Mooncake ignores @ignore_derivatives, hence the hook)
    a, b = ChainRulesCore.@ignore_derivatives ignore_derivatives_mod.(compute_chernorff_limits(mcProcess, rT, dT, T))
    n = ChainRulesCore.@ignore_derivatives 2^N
    A = (b - a) / 2
    values = fft_density_values(mcProcess, zero_rate, N, A, a, T, mode)
    dx = 2 * A / n
    return FFTDensity(values, a, a + (n - 1) * dx, dx)
end

# Density of log(S_T / S0) on the 2^N points y_j = a + 2A j / 2^N, j = 0, …, 2^N - 1: with the frequencies
# u_k = π k / A, k = -2^(N-1), …, 2^(N-1) - 1, f(y_j) ≈ Σ_k φ(-u_k) exp(i u_k a) exp(2πi jk / 2^N) / (2A), where
# exp(i u_k a) anchors the grid at its left end. The input of this transform is Hermitian (φ(u) = conj φ(-u)), and the
# result is real: the characteristic exponent is evaluated on k = 0, …, 2^(N-1) only, and transformed by brfft (the
# node -2^(N-1) through the real part of 2^(N-1), at the rounding level on the range of COS). The values are floored at
# eps.
function fft_density_values(mcProcess, r, N, A, a, T, mode)
    n2 = ChainRulesCore.@ignore_derivatives 2^(N - 1)
    FT = float_type(A)
    # -i u_k, a constant for differentiation: A comes from the range. A vector, not a range: Mooncake, which ignores
    # @ignore_derivatives, fails on the TwicePrecision of a float range.
    # the closure captures a value, not the type FT: Julia 1.11 types a captured type as DataType, and the grid as a Vector
    # of unknown element type
    zero_FT = zero(FT)
    u_im = ChainRulesCore.@ignore_derivatives adapt_array([complex(zero_FT, -k * (π / A)) for k in 0:n2], mode)
    drift_rd = FinancialMonteCarlo.integral(r.r - mcProcess.underlying.d, T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess, T)
    phase = drift_rd - corr - a
    char_exp = characteristic_exponent_i.(u_im, mcProcess, T)
    Y_r = brfft_mod((@. exp(char_exp + phase * u_im)), 2 * n2)
    twice_A = 2 * A
    density_vals_adj = density_floor.(Y_r ./ twice_A, eps(FT))
    ChainRulesCore.@ignore_derivatives @assert !isnan(sum(density_vals_adj) * twice_A / (2 * n2)) "Returned result is NaN"
    return density_vals_adj
end

# The floor gets the type of the value from the value itself: the type of a TaylorN says neither its order nor its space.
density_floor(y, eps_mod) = max(y, zero(y) + eps_mod)

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

function density_value(f::FFTDensity, y)
    v = f.values
    s = (clamp(y, f.a, f.b) - f.a) / f.dx
    i = ChainRulesCore.@ignore_derivatives stencil_index(s, length(v))
    # 1-based: the node i (0-based) is v[i + 1]
    return interpolate_density(s - i, v[i], v[i+1], v[i+2], v[i+3])
end

# Gathers: indexing with arrays of indices runs on the device, and its pullback is a scatter.
function density_array(f::FFTDensity, y::AbstractArray)
    v = f.values
    s = @. (clamp(y, f.a, f.b) - f.a) / f.dx
    # a name of its own inside the closure: assigning i there too would box it (Core.Box) and hide every type below
    i, i_1, i_2, i_3 = ChainRulesCore.@ignore_derivatives begin
        idx = stencil_index.(s, length(v))
        idx, idx .+ 1, idx .+ 2, idx .+ 3
    end
    return interpolate_density.(s .- i, v[i], v[i_1], v[i_2], v[i_3])
end
