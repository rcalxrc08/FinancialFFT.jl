module FinancialFFTTaylorSeriesExt

using FinancialFFT, TaylorSeries, FFTW
import FFTW: fft!, fft
using TaylorSeries: AbstractSeries, constant_term
import FinancialFFT: v_value_mod, abs_mod

series_order(x) = length(x.coeffs) - 1

function v_value_mod(s::Union{Taylor1, TaylorN})
    return v_value_mod(constant_term(s))
end

abs_mod(x::Union{Taylor1, TaylorN}) = signbit(v_value_mod(x)) ? -x : x

# Position of each coefficient: its degree in a Taylor1, the degree and the monomial in a TaylorN.
coefficient_indices(x::Taylor1) = 0:series_order(x)
coefficient_indices(x::TaylorN) = [(k, j) for k = 0:series_order(x) for j in eachindex(x[k].coeffs)]
get_coefficient(x::Taylor1, k) = x[k]
get_coefficient(x::TaylorN, (k, j)) = x[k][j]
set_coefficient!(x::Taylor1, c, k) = (x[k] = c)
set_coefficient!(x::TaylorN, c, (k, j)) = (x[k][j] = c)

# The fft is linear, so it maps each coefficient separately. They are written in zeros of the elements, and not in new
# series, since a TaylorN carries its JetSpace.
function fft!(x::AbstractArray{T}) where {T <: AbstractSeries{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    y = zero.(x)
    planned_fft = plan_fft!(similar(x, cpx))
    for idx in coefficient_indices(first(x))
        coeffs = [get_coefficient(el, idx) for el in x]
        planned_fft * coeffs
        for i in eachindex(y)
            set_coefficient!(y[i], coeffs[i], idx)
        end
    end
    x .= y
    nothing
end

function fft(x::AbstractArray{T}) where {T <: AbstractSeries{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    y = copy(x)
    fft!(y)
    return y
end

# Needed for interpolation. Not hasmethod: older versions of TaylorSeries reach the fallback of Base, which needs round.
which(floor, (Taylor1{Float64},)).module === TaylorSeries || (Base.floor(x::Taylor1) = floor(constant_term(x)))
which(floor, (TaylorN{Float64},)).module === TaylorSeries || (Base.floor(x::TaylorN) = floor(constant_term(x)))

end
