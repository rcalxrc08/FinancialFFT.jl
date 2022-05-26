using .TaylorSeries, FFTW
import FFTW.fft!;

function get_diff(x::AbstractArray,n::Integer)
	return TaylorSeries.getcoeff.(x,n)
end
function fft!(x::AbstractArray{T}) where {T <: AbstractSeries{cpx}} where {cpx <: Complex{num}} where {num <: Number}
	Xcomplex=[el[0] for el in x]
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
	@views idx=1:get_order(x[1])
	ders=[get_diff(x,i) for i in idx];
	for i in idx
		ders[i] .= planned_fft * ders[i]
	end
	for i in 1:length(x)
		@views ders2=[ders[order][i] for order in idx]
		@views x[i]=Taylor1([Xcomplex[i],ders2...])
	end
    nothing
end


# Needed for interpolation
import Base.floor;
!hasmethod(floor,(Taylor1,)) ? (floor(x::Taylor1) =@views floor(x[0])) : nothing
!hasmethod(floor,(AbstractSeries,)) ? (floor(x::AbstractSeries) =@views floor(x[0][1])) : nothing
