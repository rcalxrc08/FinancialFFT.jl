using .DualNumbers, FFTW
import FFTW.fft!, FFTW.fft;

function fft!(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    Xcomplex, Xder = DualNumbers.value.(x), DualNumbers.epsilon.(x)
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
    Xder .= planned_fft * Xder
    @. x = dual(Xcomplex, Xder)
    nothing
end

function fft(x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
    y = copy(x)
    fft!(y)
    return y
end

FFTW.plan_fft(x::Array{Dual{Float64}})=plan_fft(1:length(x))
import Base.*
function *(planned_fft::FFTW.cFFTWPlan,x::Array{Dual{Float64}})
	Xcomplex, Xder = DualNumbers.value.(x), DualNumbers.epsilon.(x)
    Xcomplex_n = planned_fft * Xcomplex
    Xder_n = planned_fft * Xder
    return dual.(Xcomplex_n, Xder_n)
end
function *(planned_fft::FFTW.cFFTWPlan,x::AbstractArray{T}) where {T <: Dual{cpx}} where {cpx <: Complex{num}} where {num <: Number}
	Xcomplex, Xder = DualNumbers.value.(x), DualNumbers.epsilon.(x)
    Xcomplex .= planned_fft * Xcomplex
    Xder .= planned_fft * Xder
    return dual.(Xcomplex, Xder)
end