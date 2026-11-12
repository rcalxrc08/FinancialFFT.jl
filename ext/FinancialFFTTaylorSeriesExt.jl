module FinancialFFTTaylorSeriesExt

using FinancialFFT, TaylorSeries, FFTW
import FFTW: fft!, fft
using TaylorSeries: AbstractSeries, constant_term
import FinancialFFT: v_value_mod, brfft_mod, zero_of_type

series_order(x) = length(x.coeffs) - 1

function v_value_mod(s::Union{Taylor1, TaylorN})
    return v_value_mod(constant_term(s))
end

# The zero of a series has the order of the values: their product (the type of a series does not say its order).
zero_of_type(::Type{<:Union{Taylor1, TaylorN}}, x...) = zero(prod(x))

# Position of each coefficient: its degree in a Taylor1, the degree and the monomial in a TaylorN.
coefficient_indices(x::Taylor1) = 0:series_order(x)
coefficient_indices(x::TaylorN) = [(k, j) for k = 0:series_order(x) for j in eachindex(x[k].coeffs)]
get_coefficient(x::Taylor1, k) = x[k]
get_coefficient(x::TaylorN, (k, j)) = x[k][j]
set_coefficient!(x::Taylor1, c, k) = (x[k] = c)
set_coefficient!(x::TaylorN, c, (k, j)) = (x[k][j] = c)

# The fft is linear, so it maps each coefficient separately: one buffer for the coefficients of every degree, transformed
# in place with one plan. They are written in zeros of the elements, and not in new series, since a TaylorN carries its
# JetSpace; nor in the elements of x, which may be shared (by fill, or by the shallow copy of fft).
function fft_coefficients(x)
    y = zero.(x)
    indices = coefficient_indices(first(x))
    buffer = [get_coefficient(el, first(indices)) for el in x]
    planned_fft = FinancialFFT.plan_fft_mod!(buffer)
    for idx in indices
        map!(el -> get_coefficient(el, idx), buffer, x)
        planned_fft * buffer
        for i in eachindex(y)
            set_coefficient!(y[i], buffer[i], idx)
        end
    end
    return y
end

function fft!(x::AbstractArray{T}) where {T <: AbstractSeries{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    x .= fft_coefficients(x)
    nothing
end

fft(x::AbstractArray{T}) where {T <: AbstractSeries{cpx}} where {cpx <: Complex{num}} where {num <: Number} = fft_coefficients(x)

# brfft is linear too, coefficient by coefficient, with one plan and one buffer for the input and one for the output;
# its output has n real series, built as zeros of the real part of an element so that they keep its order (and JetSpace).
function brfft_mod(x::AbstractVector{T}, n::Integer) where {T <: AbstractSeries{<:Complex}}
    y = [zero(real(first(x))) for _ in 1:n]
    indices = coefficient_indices(first(x))
    buffer = [get_coefficient(el, first(indices)) for el in x]
    plan = plan_brfft(buffer, n)
    coeffs = similar(buffer, real(eltype(buffer)), n)
    for idx in indices
        map!(el -> get_coefficient(el, idx), buffer, x)
        FinancialFFT.mul_mod!(coeffs, plan, buffer)
        for i in eachindex(y)
            set_coefficient!(y[i], coeffs[i], idx)
        end
    end
    return y
end

# Needed for interpolation. Not hasmethod: older versions of TaylorSeries reach the fallback of Base, which needs round.
which(floor, (Taylor1{Float64},)).module === TaylorSeries || (Base.floor(x::Taylor1) = floor(constant_term(x)))
which(floor, (TaylorN{Float64},)).module === TaylorSeries || (Base.floor(x::TaylorN) = floor(constant_term(x)))

end
