using FFTW, .ForwardDiff
import FFTW.fft!;

function fft!(x::AbstractArray{T1}) where {T1 <: Complex{cpx}} where {cpx <: ForwardDiff.Dual{Tg, T, N}} where {Tg, T <: Real, N}
    Xcomplex = [ForwardDiff.value(real(el)) + 1im * ForwardDiff.value(imag(el)) for el in x]
    Xder1 = reduce(hcat, [ForwardDiff.partials(real(el)) + 1im * ForwardDiff.partials(imag(el)) for el in x])'
    planned_fft = plan_fft!(similar(Xcomplex))
    Xcomplex .= planned_fft * Xcomplex
    for col in eachcol(Xder1)
        col .= planned_fft * col
    end
    npartials = size(Xder1, 2)
    epss = [ForwardDiff.Dual{Tg}(0, ntuple(i -> i == k, Val(npartials))) for k = 1:npartials]
    x .= Xcomplex .+ Xder1 * epss
    nothing
end

function fft(x::AbstractArray{T1}) where {T1 <: Complex{cpx}} where {cpx <: ForwardDiff.Dual{Tg, T, N}} where {Tg, T <: Real, N}
    y = copy(x)
    fft!(y)
    return y
end