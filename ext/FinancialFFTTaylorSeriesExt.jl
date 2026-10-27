module FinancialFFTTaylorSeriesExt

using FinancialFFT, TaylorSeries, FFTW
import FFTW: fft!, fft
using TaylorSeries: AbstractSeries
import FinancialFFT: v_value_mod

series_order(x::Taylor1) = length(x.coeffs) - 1

function v_value_mod(s::Taylor1)
    return v_value_mod(s[0])
end

function get_diff(x::AbstractArray, n::Integer)
    return TaylorSeries.getcoeff.(x, n)
end

function fft!(x::AbstractArray{T}) where {T <: AbstractSeries{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex = [el[0] for el in x]
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
    idx = 1:series_order(x[1])
    ders = [get_diff(x, i) for i in idx]
    for i in idx
        ders[i] .= planned_fft * ders[i]
    end
    for i in eachindex(x)
        @views ders2 = [ders[order][i] for order in idx]
        @views x[i] = Taylor1([Xcomplex[i], ders2...])
    end
    nothing
end

function fft(x::AbstractArray{T}) where {T <: AbstractSeries{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    y = copy(x)
    fft!(y)
    return y
end

# Needed for interpolation
!hasmethod(floor, (Taylor1,)) ? (Base.floor(x::Taylor1) = @views floor(x[0])) : nothing
!hasmethod(floor, (AbstractSeries,)) ? (Base.floor(x::AbstractSeries) = @views floor(x[0][1])) : nothing

end
